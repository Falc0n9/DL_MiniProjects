from torch import split, Tensor, cat, LongTensor
from torch.autograd import Variable
import numpy
import dlc_bci as bci
from math import floor
import numpy

#train_input, train_target = bci.load(root = './data_bci')
#test_input, test_target = bci.load(root = './data_bci', train = False)
#validate_size = 52
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

