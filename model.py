import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import random
import argparse

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import aug_functions



def model_nvidia_2(train, train_samples, epochs, validation, valid_samples, verbose, destination):
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (66,200,3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode = "valid", init = 'he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
    model.add(Dropout(0.3))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode = "valid", init = 'he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
    model.add(Dropout(0.3))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode = "valid", init = 'he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode = "valid", init = 'he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
    model.add(Dropout(0.3))
#     model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode = "valid", init = 'he_normal'))
#     model.add(ELU())

#     model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
#     model.add(Dense(1164, init='he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
#     model.add(Dropout(0.3))
    model.add(Dense(100, init='he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
#     model.add(Dropout(0.3))
    model.add(Dense(50, init='he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
#     model.add(Dropout(0.3))
    model.add(Dense(10, init='he_normal'))
#     model.add(Activation('relu'))
    model.add(ELU())
#     model.add(Dropout(0.3))
    model.add(Dense(1, init='he_normal'))
    
    model.compile(loss = 'mse', optimizer = 'adam')
#     model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)
    model.fit_generator(train, samples_per_epoch = train_samples, nb_epoch = epochs, validation_data = validation, nb_val_samples = valid_samples, verbose=verbose)
    model.save(destination)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Model Generator")
	parser.add_argument('directory', type=str, help="Path to data folder to train model")
	parser.add_argument('epochs', type=int, default=5, help="Number of epochs to train")
	parser.add_argument('train_samples', type=int, default=50000, help="Number of train samples")
	parser.add_argument('valid_samples', type=int, default=17000, help="Number of validation samples")
	parser.add_argument('verbose', type=int, default=2, help="Set verbose flag")
	parser.add_argument('destination', type=str, default='models/model.h5', help="Save file name")
	args = parser.parse_args()


	generate_batch = aug_functions.train_pipe(args.directory)
	generate_valid = aug_functions.valid_pipe(args.directory)

	model_nvidia_2(generate_batch, args.train_samples, args.epochs, generate_valid, args.valid_samples, args.verbose, args.destination)
