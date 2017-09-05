from parameters import *
from traffic_sign.RandomTransform import *


class OptTransform:

    def __init__(self, model, target=True, batch_size=BATCH_SIZE, n_step=1000,
                 c=10, lr=0.01, init_scl=0.5, use_bound=False):

        self.model = model
        self.n_step = n_step
        self.batch_size = batch_size

        # Initialize variables
        init_val = np.random.normal(scale=init_scl, size=((1,) + INPUT_SHAPE))
        #init_val = np.random.uniform(high=scale, low=-scale, size=((1,) + INPUT_SHAPE))
        self.d = tf.Variable(initial_value=init_val, trainable=True,
                             dtype=tf.float32)
        self.x_orig = K.placeholder(
            dtype='float32', shape=((self.batch_size,) + INPUT_SHAPE))
        self.y_orig = K.placeholder(
            dtype='float32', shape=(self.batch_size, OUTPUT_DIM))
        x_in = self.x_orig + self.d

        # Calculate loss
        loss_all = K.categorical_crossentropy(
            self.y_orig, self.model(x_in), from_logits=False)
        self.loss = tf.reduce_sum(loss_all)
        self.loss /= self.batch_size
        if not target:
            self.loss *= -1

        # Regularization term with l2-norm
        self.norm = tf.norm(self.d, ord='euclidean')
        # Objective function
        self.f = c * self.norm + self.loss
        # Setup optimizer
        if use_bound:
            # Use Scipy optimizer
            self.optimizer = ScipyOptimizerInterface(self.f, var_list=[self.d],
                                                     var_to_bounds={
                                                         x_in: (0, 1)},
                                                     method="L-BFGS-B")
        else:
            # Use Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                                    beta2=0.999, epsilon=1e-08)
            self.opt = self.optimizer.minimize(self.f, var_list=[self.d])

        # Initialize transformer
        seed = np.random.randint(1234)
        self.rnd_transform = RandomTransform(seed=seed, p=1.0, intensity=0.3)

    def optimize(self, x, y):

        with tf.Session() as sess:

            # Initialize variables and load weights
            sess.run(tf.global_variables_initializer())
            self.model.load_weights(WEIGTHS_PATH)

            # Generate transformed images here if all batches should be same
            x_ = np.zeros((self.batch_size,) + INPUT_SHAPE)
            y_ = np.repeat([y], self.batch_size, axis=0)
            x_[0] = np.copy(x)
            for i in range(1, self.batch_size):
                x_[i] = self.rnd_transform.transform(x)
            # TODO: Apply mask

            feed_dict = {self.x_orig: x_, self.y_orig: y_,
                         K.learning_phase(): False}

            for step in range(self.n_step):
                print sess.run(self.norm)
                print sess.run(self.loss, feed_dict=feed_dict)
                print sess.run(self.f, feed_dict=feed_dict)
                sess.run(self.opt, feed_dict=feed_dict)
            return (x + sess.run(self.d)).reshape(INPUT_SHAPE)
