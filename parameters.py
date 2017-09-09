# Import packages for all files
import random
import threading
import time
import os
from os import listdir

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import keras.backend as K
from scipy import misc
import cv2
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import pickle


# Set constants
NUM_LABELS = 43                             # Number of labels
BATCH_SIZE = 32                             # Size of batch
HEIGHT = 32
WIDTH = 32
N_CHANNEL = 3                               # Number of channels
OUTPUT_DIM = 43                             # Number of output dimension
NUM_EPOCH = 100                             # Number of epoch to train

# Set paths
WEIGTHS_PATH = "./keras_weights/weights_mltscl.hdf5"   # Path to saved weights
DATA_DIR = "./input_data/GTSRB/"        # Path to directory containing dataset

INPUT_SHAPE = (HEIGHT, WIDTH, N_CHANNEL)    # Input shape of model
IMAGE_SIZE = (HEIGHT, WIDTH)                # Height and width of resized image
N_FEATURE = HEIGHT * WIDTH * N_CHANNEL      # Number of input dimension
