from lib.keras_utils import *
from lib.RandomTransform import *
from lib.utils import *
from parameters import *

EPS = 1e-10   # Epsilon
MIN_CP = -2.  # Minimum power index of c
MAX_CP = 2.   # Maximum power index of c
SCORE_THRES = 0.5  # Softmax score threshold to consider success of attacks
PROG_PRINT_STEPS = 50  # Print progress every certain steps
EARLYSTOP_STEPS = 5000  # Early stopping if no improvement for certain steps


class OptTransform:
    """
    This class implements a generator for adversarial examples that are robust
    to certain transformations or variations. It is a modification from 
    Carlini et al. (https://arxiv.org/abs/1608.04644) and Athalye et al. 
    (https://arxiv.org/abs/1707.07397)
    """

    def _setup_opt(self):
        """Used to setup optimization when c is updated"""

        # obj_func = c * loss + l2-norm(d)
        self.f = self.c * self.loss + self.norm
        # Setup optimizer
        if self.use_bound:
            # Use Scipy optimizer with upper and lower bound [0, 1]
            self.optimizer = ScipyOptimizerInterface(
                self.f, var_list=self.var_list, var_to_bounds={
                    self.x_in: (0, 1)},
                method="L-BFGS-B")
        else:
            # Use learning rate decay
            global_step = tf.Variable(0, trainable=False)
            # decay_lr = tf.train.exponential_decay(
            #   self.lr, global_step, 500, 0.95, staircase=True)
            decay_lr = tf.train.inverse_time_decay(
                self.lr, global_step, 10, 0.01)
            # Use Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=decay_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.opt = self.optimizer.minimize(
                self.f, var_list=self.var_list, global_step=global_step)

    def __init__(self, model, target=True, c=1, lr=0.01, init_scl=0.1,
                 use_bound=False, loss_op=0, k=5, var_change=True,
                 use_mask=True, batch_size=BATCH_SIZE):
        """
        Initialize the optimizer. Default values of the parameters are
        recommended and decided specifically for attacking traffic sign
        recognizer trained on GTSRB dataset.

        Parameters
        ----------
        model      : Keras model
                     Target model to attack
        target     : (optional) bool (default: True)
                     True if performing targeted attack; False, otherwise.
        c          : (optional) float (default: 1)
                     Constant balancing the objective function (f) between norm
                     of perturbation and loss (f = c * loss + norm). Larger c
                     means stronger attack but also more "visible" (stronger
                     perturbation).
        lr         : (optional) float (default: 0.01)
                     Learning rate of optimizer
        init_scl   : (optional) float (default: 0.1)
                     Standard deviation of Gaussian used to initialize objective
                     variable
        use_bound  : (optional) bool (default: False)
                     If True, optimizer with bounding box [0, 1] will be used.
                     Otherwise, Adam optimizer is used.
        loss_op    : (optional) int (default: 0)
                     Option for loss function to optimize over.
                     loss_op = 0: Carlini's l2-attack
                     loss_op = 1: cross-entropy loss
        k          : (optional) float (default: 5)
                     "Confidence threshold" used with loss_op = 0. Used to
                     control strength of attack. The higher the k the stronger
                     the attack.
        var_change : (optional) bool (default: True)
                     If True, objective variable will be changed according to
                     Carlini  et al. (which also gets rid of the need to use
                     any bounding) Otherwise, optimize directly on perturbation.
        use_mask   : (optional) bool (default: True)
                     if True, perturbation will be masked before applying to
                     the target sign. Mask must be specified when calling
                     optimize() and optimize_search().
        batch_size : (optional) int (default: BATCH_SIZE)
                     Define number of transformed images to use
        """

        self.model = model
        self.target = target
        self.c = c
        self.lr = lr
        self.use_bound = use_bound
        self.loss_op = loss_op
        self.k = k
        self.use_mask = use_mask
        self.batch_size = batch_size

        # Initialize variables
        init_val = np.random.normal(scale=init_scl, size=((1,) + INPUT_SHAPE))
        self.x_orig = K.placeholder(
            dtype='float32', shape=((1,) + INPUT_SHAPE))
        self.x = K.placeholder(
            dtype='float32', shape=((self.batch_size,) + INPUT_SHAPE))
        self.y = K.placeholder(
            dtype='float32', shape=(self.batch_size, OUTPUT_DIM))
        if self.use_mask:
            self.m = K.placeholder(dtype='float32', shape=((1,) + INPUT_SHAPE))

        # If change of variable is specified
        if var_change:
            # Optimize on w instead of d
            self.w = tf.Variable(initial_value=init_val, trainable=True,
                                 dtype=tf.float32)
            x_full = (0.5 + EPS) * (tf.tanh(self.w) + 1)
            self.d = x_full - self.x_orig
            if self.use_mask:
                self.d = tf.multiply(self.d, self.m)
            self.x_in = self.x + self.d
            self.var_list = [self.w]
        else:
            # Optimize directly on d (perturbation)
            self.d = tf.Variable(initial_value=init_val, trainable=True,
                                 dtype=tf.float32)
            if self.use_mask:
                self.d = tf.multiply(self.d, self.m)
            self.x_in = self.x + self.d
            self.var_list = [self.d]

        model_output = self.model(self.x_in)

        if loss_op == 0:
            # Carlini l2-attack's loss
            # Get 2 largest outputs
            output_2max = tf.nn.top_k(model_output, 2)[0][0]
            # Find z_i = max(Z[i != y])
            i_y = tf.argmax(self.y, axis=1, output_type=tf.int32)[0]
            i_max = tf.to_int32(tf.argmax(model_output, axis=1)[0])
            z_i = tf.cond(tf.equal(i_y, i_max),
                          lambda: output_2max[1], lambda: output_2max[0])
            if self.target:
                # loss = max(max(Z[i != t]) - Z[t], -K)
                self.loss = tf.maximum(z_i - model_output[0, i_y], -self.k)
            else:
                # loss = max(Z[y] - max(Z[i != y]), -K)
                self.loss = tf.maximum(model_output[0, i_y] - z_i, -self.k)
        elif loss_op == 1:
            # Cross entropy loss, loss = -log(F(x_t))
            loss_all = K.categorical_crossentropy(
                self.y, model_output, from_logits=True)
            self.loss = tf.reduce_sum(loss_all)
            self.loss /= self.batch_size
            if not self.target:
                # loss = log(F(x_y))
                self.loss *= -1
        else:
            raise ValueError("Invalid loss_op")

        # Regularization term with l2-norm
        self.norm = tf.norm(self.d, ord='euclidean')
        self._setup_opt()

        # Initialize transformer
        seed = np.random.randint(1234)
        self.rnd_transform = RandomTransform(seed=seed, p=1.0, intensity=0.3)

    def optimize(self, x, y, n_step=1000, prog=True, mask=None):

        with tf.Session() as sess:

            # Initialize variables and load weights
            sess.run(tf.global_variables_initializer())
            self.model.load_weights(WEIGTHS_PATH)

            # Create inputs to optimization
            x_ = np.zeros((self.batch_size,) + INPUT_SHAPE)
            x_orig_ = np.copy(x).reshape((1,) + INPUT_SHAPE)
            y_ = np.repeat([y], self.batch_size, axis=0)

            # Generate a batch of transformed images
            x_[0] = np.copy(x)
            for i in range(1, self.batch_size):
                x_[i] = self.rnd_transform.transform(x)

            # Include mask in feed_dict if mask is used
            if self.use_mask:
                m_ = np.repeat(
                    mask[np.newaxis, :, :, np.newaxis], N_CHANNEL, axis=3)
                feed_dict = {self.x: x_, self.y: y_, self.x_orig: x_orig_,
                             self.m: m_, K.learning_phase(): False}
            else:
                feed_dict = {self.x: x_, self.y: y_, self.x_orig: x_orig_,
                             K.learning_phase(): False}

            # Set up some variables for early stopping
            min_norm = float("inf")
            min_d = None
            earlystop_count = 0

            # Start optimization
            for step in range(n_step):
                if self.use_bound:
                    self.optimizer.minimize(sess, feed_dict=feed_dict)
                else:
                    sess.run(self.opt, feed_dict=feed_dict)

                # Keep track of "best" solution
                if self.loss_op == 0:
                    norm = sess.run(self.norm, feed_dict=feed_dict)
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    # Save working adversarial example with smallest norm
                    if loss == -self.k:
                        if norm < min_norm:
                            min_norm = norm
                            min_d = sess.run(self.d, feed_dict=feed_dict)
                            # Reset early stopping counter
                            earlystop_count = 0
                        else:
                            earlystop_count += 1
                            # Early stop if no improvement
                            if earlystop_count > EARLYSTOP_STEPS:
                                print step, min_norm
                                break

                # Print progress
                if (step % PROG_PRINT_STEPS == 0) and prog:
                    f = sess.run(self.f, feed_dict=feed_dict)
                    norm = sess.run(self.norm, feed_dict=feed_dict)
                    loss = sess.run(self.loss, feed_dict=feed_dict)
                    print "Step: {}, norm={:.3f}, loss={:.3f}, obj={:.3f}".format(step, norm, loss, f)

            if min_d is not None:
                x_adv = (x_orig_ + min_d).reshape(INPUT_SHAPE)
                return x_adv, min_norm
            else:
                d = sess.run(self.d, feed_dict=feed_dict)
                norm = sess.run(self.norm, feed_dict=feed_dict)
                x_adv = (x_orig_ + d).reshape(INPUT_SHAPE)
                return x_adv, norm
