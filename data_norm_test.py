from mini_project_1 import *
from torch import optim, unsqueeze, div
import dlc_bci as bci
from helperfunctions import *
from torch.autograd import Variable
from random import randint

train_input_base, train_target_base = bci.load(root='./data_bci')
test_input, test_target = bci.load(root='./data_bci', train=False)

validate_size = 50
train_input, train_target, validate_input, validate_target = cross_val_datasets(train_input_base, train_target_base,
                                                                                validate_size)


# Making each dataset Variable for allowing autograd functionality.
validate_input, validate_target, train_target, train_input = Variable(validate_input), Variable(
    validate_target), Variable(train_target), Variable(train_input)
test_target, test_input = Variable(test_target), Variable(test_input)

for i in range(train_input.size(0)):
    mu, std = train_input.data[i].mean(2).mean(0), train_input.data[i].std(2).std(0)
    mu = unsqueeze(unsqueeze(mu, 0), 2)
    std = unsqueeze(unsqueeze(std, 0), 2)
    print(mu.size())
    print(std.size())
    train_input.data[i] = (train_input.data[i]-mu)
    train_input.data[i] = div(train_input.data[i], std)
    print(train_input.data[i].size())