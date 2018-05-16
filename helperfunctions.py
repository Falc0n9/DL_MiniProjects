import dlc_bci as bci
import numpy
from dlc_practical_prologue import convert_to_one_hot_labels
from math import ceil, floor, sqrt, pi
from torch import cuda, nn, optim, split, Tensor, cat, LongTensor
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets
import matplotlib.pyplot as plt


'''
Compute_nb_errors calculates the amount of wrongly predicted labels by the model
'''
def compute_nb_errors(model, input, target, mini_batch_size):

    nb_errors = 0

    for b in range(0, input.size(0) - input.size(0)%mini_batch_size, mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(0, mini_batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

'''
Plot_data is used in mini_project_2 to plot the datapoints and apoint a color to them according to their label. 
Green stands for label 1 meaning the dot should be in the circle with radius 1/(2*Pi) and red for label 0.
'''
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


'''
cross_val_datasets is used to split up the original dataset into multiple training and validation sets.
The length of the validation set can be chosen. The function will iterate through the dataset and make a series of unique validation sets with equal length.
The training sets are the remaining datapoints in the dataset. 
'''
def cross_val_datasets(input, target, validate_size):
    new_length = floor(input.size(0)/validate_size)*validate_size
    validate_input_tuple = split(input[0: new_length], validate_size, 0)
    validate_target_tuple = split(target[0: new_length], validate_size, 0)
    validate_input = Tensor()
    validate_target = LongTensor()
    train_input = Tensor()
    train_target = LongTensor()

    for i in range(len(validate_input_tuple)):
        validate_input = cat((validate_input, validate_input_tuple[i]))
        validate_target = cat((validate_target, validate_target_tuple[i].long()))
        for n in range(len(validate_input_tuple)):
            if i != n:
                train_input = cat((train_input,validate_input_tuple[n]),0)
                train_target = cat((train_target,validate_target_tuple[n].long()),0)  

    validate_input = validate_input.view(len(validate_input_tuple),validate_size,train_input.size(1), train_input.size(2))
    validate_target = validate_target.view(len(validate_input_tuple),validate_size)
    train_input = train_input.view(len(validate_input_tuple),new_length-validate_size, train_input.size(1), train_input.size(2))
    train_target = train_target.view(len(validate_target_tuple), new_length-validate_size)

    return train_input, train_target, validate_input, validate_target

