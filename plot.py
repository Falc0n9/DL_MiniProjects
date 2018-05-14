# This function allows to plot the data and results from Mini Project 2
import numpy
from math import sqrt, pi
import matplotlib.pyplot as plt


def plot_data(train_data,train_target):
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    train_data_numpy =  train_data.numpy()
    train_target_numpy = train_target.numpy()
    train_data_numpy_x = [train_data_numpy[i][0] for i in range(len(train_data_numpy))]
    train_data_numpy_y = [train_data_numpy[i][1] for i in range(len(train_data_numpy))]
    train_data_numpy_x_in = train_data_numpy_x*train_target_numpy
    train_data_numpy_y_in = train_data_numpy_y*train_target_numpy
    train_data_numpy_x_out = train_data_numpy_x-train_data_numpy_x_in
    train_data_numpy_y_out = train_data_numpy_y-train_data_numpy_y_in

    plt.plot(train_data_numpy_x_in,train_data_numpy_y_in,'g.')
    plt.plot(train_data_numpy_x_out,train_data_numpy_y_out,'r.')

    circle = plt.Circle((0, 0), 1/sqrt(2*pi), color='r', fill=False)
    ax.add_artist(circle)

    #plt.show()


    #Plotting Data
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    train_data_numpy =  train_data.numpy()
    train_target_numpy = train_target.numpy()
    train_data_numpy_x = [train_data_numpy[i][0] for i in range(len(train_data_numpy))]
    train_data_numpy_y = [train_data_numpy[i][1] for i in range(len(train_data_numpy))]
    train_data_numpy_x_in = train_data_numpy_x*train_target_numpy
    train_data_numpy_y_in = train_data_numpy_y*train_target_numpy
    train_data_numpy_x_out = train_data_numpy_x-train_data_numpy_x_in
    train_data_numpy_y_out = train_data_numpy_y-train_data_numpy_y_in

    plt.plot(train_data_numpy_x_in,train_data_numpy_y_in,'g.')
    plt.plot(train_data_numpy_x_out,train_data_numpy_y_out,'r.')

    circle = plt.Circle((0, 0), 1/sqrt(2*pi), color='k',linewidth = 6.0, fill=False)
    ax.add_artist(circle)

    plt.show()