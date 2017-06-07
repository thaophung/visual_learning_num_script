#!/usr/bin/env python
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Layer
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle

batch_size = 100
#batch_size = 20
nb_epoch = 20

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--L2', metavar='w', type=float, default=0.0, nargs='?', help='L2 weight')
    parser.add_argument('--lr', metavar='lr', type=float, default=0.001, nargs='?', help='Learning rate')
    parser.add_argument('--seed', metavar='n', type=int, default=1337, nargs='?', help='learning rate')

    args = parser.parse_args()

    np.random.seed(args.seed)  # for reproducibility

    # which neuron to visualize
    print "-----------------"
    print " L2: %s" % args.L2
    print " Learning rate: %s" % args.lr
    print " seed: %s" % args.seed
    print "-----------------"


    # Load dataset
    train_X = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    test_X = np.load('test_X.npy')
    test_y = np.load('test_y.npy')


    # Netework
    model = Sequential()
    model.add(Flatten(input_shape=(15,60,2)))
    model.add(Dense(128, kernel_regularizer=l2(args.L2)))
    model.add(Activation('relu'))
    model.add(Dense(128, kernel_regularizer=l2(args.L2)))
    model.add(Activation('relu'))
    model.add(Dense(128, kernel_regularizer=l2(args.L2)))
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_regularizer=l2(args.L2)))
    
    print model.summary()

    adam = Adam(args.lr)
    #adagrad = Adagrad(lr=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['binary_accuracy'])

    history = model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(test_X, test_y))
    model.save_weights('model.h5', overwrite=True)
    
    log_file = "%s_%s_%s" % (
            args.L2,
            args.lr,
            args.seed
    )

    # list all data in history
    print(history.history.keys())
    loss_acc = open(log_file + ".log", "wb")
    pickle.dump(history.history, loss_acc)

if __name__ == "__main__":
	main()
