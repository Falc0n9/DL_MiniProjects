from mini_project_1 import *
from torch import optim, unsqueeze, div, Tensor
import dlc_bci as bci
from helperfunctions import *
from torch.autograd import Variable
from random import randint

def train_model(model,
                train_input, train_target,
                criterion=nn.CrossEntropyLoss, optimizer=optim.SGD,
                lr=0.1, nb_epochs=100, batch_size=10,
                L1_penalty=False, L2_penalty=False,
                lambda_L1=0.0001, lambda_L2=0.0001):
    """
    :param model: The model to train
    :param train_input: Training input
    :param train_target: Training targets
    :param criterion: The loss criterion
    :param optimizer: The training optimizer
    :param lr: The learning rate
    :param nb_epochs: The number of epochs
    :param batch_size: The batch-size
    :param L1_penalty: Whether to apply L1 penalty
    :param L2_penalty: Whether to apply L2 penalty
    :param lambda_L1: Lambda for L1 penalty
    :param lambda_L2: Lambda for L2 penalty
    """

    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr=lr)

    # Normalizing data
    mu, std = train_input.data.mean(), train_input.data.std()
    train_input.data.sub_(mu).div_(std)

    for k in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0) - train_input.size(0) % batch_size, batch_size):

            output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(output, train_target.narrow(0, b, batch_size))

            # Implements L2 penalization
            if L2_penalty:
                for p in model.parameters():
                    loss += lambda_L2 * p.pow(2).sum()

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Implements L1 penalization
            if L1_penalty:
                for p in model.parameters():
                    p.data -= p.data.sign() * p.data.abs().clamp(max=lambda_L1)

            sum_loss = sum_loss + loss.data[0]


# Loading dataset
train_input_base, train_target_base = bci.load(root='./data_bci')
test_input, test_target = bci.load(root='./data_bci', train=False)

# Creation of cross validation datasets with size 'validate_size'.
validate_size = 50
train_input, train_target, validate_input, validate_target = cross_val_datasets(train_input_base, train_target_base,
                                                                                validate_size)

# Making each dataset Variable for allowing autograd functionality.
validate_input, validate_target, train_target, train_input = Variable(validate_input), Variable(
    validate_target), Variable(train_target), Variable(train_input)
test_target, test_input = Variable(test_target), Variable(test_input)

# Training parameters
lr, nb_epochs, batch_size = 0.01, 50, 30
lambda_L1 = lambda_L2 = 0.0001

conv_layer = [(3, 40),(3,20)]
linear_layer = [20]

# Cross-validation loop
train_acc_avg = 0
val_acc_avg = 0
test_acc_avg = 0

for i in range(train_input.size(0)):
    model = Net(conv_layer,linear_layer, act_func=ReLU, with_dropout_conv=False, with_batchnorm_conv=False, with_dropout_lin=False, with_batchnorm_lin=False)


    # Model training
    model.train(True)
    train_model(model, train_input[i], train_target[i], criterion=nn.CrossEntropyLoss, optimizer=optim.SGD, lr=lr, L1_penalty= False, L2_penalty=True)
    model.train(False)
    train_acc = 100 * (1 - compute_nb_errors(model, train_input[i], train_target[i], batch_size) / len(train_input[i]))
    val_acc = 100 * (1 - compute_nb_errors(model, validate_input[i], validate_target[i], batch_size) / len(validate_input[i]))
    test_acc = 100*(1 - compute_nb_errors(model, test_input, test_target, batch_size) / len(test_input))
    train_acc_avg += train_acc
    val_acc_avg += val_acc
    test_acc_avg += test_acc
    
    # Report results
    print(i, " Train Accuracy:", train_acc)
    print(i, " Validate Accuracy:", val_acc)
    print(i, " Test Accuracy:", test_acc)
    print("-------------------------------------------------------------")

print("Avg. Train acc.:", train_acc_avg/train_input.size(0))
print("Avg. Validate acc.:", val_acc_avg/train_input.size(0))
print("Avg. Test acc.:", test_acc_avg/train_input.size(0))


