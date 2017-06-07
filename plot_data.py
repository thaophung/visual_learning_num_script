import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def plot_acc(history, name):
    plt.plot(history['binary_accuracy'])
    plt.plot(history['val_binary_accuracy'])
    plt.title(name + "_accuracy")
    plt.ylabel('accuracy')
    plt.ylim([0,1])
    plt.xlabel('epoch')
    plt.xlim([0,len(history['binary_accuracy'])])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name + "_acc.jpg")
    plt.close()

def plot_loss(history, name):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(name+'_loss')
    plt.ylabel('loss')
    plt.ylim([0, 12000])
    plt.xlabel('epoch')
    plt.xlim([0,len(history['loss'])])
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+'_loss.jpg')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', metavar='n', type=str, default="a", nargs='?', help='Plot file name')

    args = parser.parse_args()

    filename = args.filename
    loss_acc_file = open(filename,'rb')
    loss_acc_file = pickle.load(loss_acc_file)

    plot_acc(loss_acc_file, filename)
    plot_loss(loss_acc_file, filename)

if __name__ == "__main__":
    main()
