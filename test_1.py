from mini_project_1 import *
from torch import optim
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
learning_rates, nb_epochs, batch_size = [1,0.1,0.001,0.0001], 100, 10
lambda_L1 = lambda_L2 = 0.0001
act_funcs = [ReLU(), Tanh()]

for i in range(len(train_input)):
    for sigma in act_funcs:
        for lr in learning_rates:
            linear_layer = []
            for nb_conv_layers in range(1,5):
                for ks in range(2,8):
                    for nb_ch in range(1,30):
                        conv_layer = []
                        for i in range(nb_conv_layers):
                            conv_layer += [(ks, nb_ch)] 
                        
                        nb_linear_layers = randint(1,4)
                        for j in range(nb_linear_layers):
                            hidden_units = randint(4,500)
                            linear_layer += [hidden_units]
            # Cross-validation loop
                        for i in range(train_input.size(0)):
                            model = Net(conv_layer,linear_layer, with_dropout_conv=False, with_dropout_lin=True)


                            # Model training
                            model.train(True)
                            train_model(model, train_input[i], train_target[i], optimizer=optim.Adadelta)
                            model.train(False)

                            # Report results
                            print(i, " Train Accuracy:",
                                100 * (1 - compute_nb_errors(model, train_input[i], train_target[i], batch_size) / len(train_input[i])))
                            print(i, " Validate Accuracy:", 100 * (
                                    1 - compute_nb_errors(model, validate_input[i], validate_target[i], batch_size) / len(validate_input[i])))
                            #print(i, " Test Accuracy:", 100 * (
                            #        1 - compute_nb_errors(model, test_input, test_target, batch_size) / len(test_input)))
                            print("-------------------------------------------------------------")




