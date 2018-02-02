import keras
from keras import backend as K
from tensorflow.python.platform import flags
from keras.models import save_model

from tf_utils import tf_train, tf_test_error_rate
from lib.attacks import symbolic_fgs, symb_iter_fgs
from os.path import basename

from lib.keras_utils import *
from lib.utils import *
from parameters import *

FLAGS = flags.FLAGS


def main():
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    flags.DEFINE_bool('NUM_EPOCHS', args.epochs, 'Number of epochs')

    # Get MNIST test data
    TRAIN_FILE_NAME = 'train_extended_75.p'

    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_GTSRB(
    n_channel=N_CHANNEL, train_file_name=TRAIN_FILE_NAME)

    x = K.placeholder(shape=(None,
                             HEIGHT,
                             WIDTH,
                             N_CHANNEL))

    y = K.placeholder(shape=(BATCH_SIZE, NUM_LABELS))

    eps = args.eps
    norm = args.norm

    x_advs = [None]

    model = build_mltscl_adv(x)

    if args.iter == 0:
        logits = model.output
        grad = gen_grad(x, logits, y, loss='training')
        x_advs = symbolic_fgs(x, grad, eps=eps)
    elif args.iter == 1:
        x_advs = symb_iter_fgs(m, x, y, steps = 40, alpha = 0.01, eps = args.eps)

    # Train an MNIST model
    tf_train(x, y, model, X_train, Y_train, data_gen, x_advs=x_advs, benign = args.ben)

    # Finally print the result!
    test_error = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % test_error)
    model_name += '_' + str(eps) + '_' + str(norm) + '_' + ens_str
    if args.iter == 1:
        model_name += 'iter'
    if args.ben == 0:
        model_name += '_nob'

    model_name = 'multiscale_adv'
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'wr') as f:
        f.write(json_string)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12,
                        help="number of epochs")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--norm", type=str, default='linf',
                        help="norm used to constrain perturbation")
    parser.add_argument("--iter", type=int, default=0,
                        help="whether an iterative training method is to be used")
    parser.add_argument("--ben", type=int, default=1,
                        help="whether benign data is to be used while performing adversarial training")

    args = parser.parse_args()
    main()
