import dlc_bci as bci
from math import ceil, floor
from torch import cuda, nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets
from dlc_practical_prologue import convert_to_one_hot_labels
from helperfunctions import compute_nb_errors, cross_val_datasets

#Defining structure of neural network in class Net()
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        #Defining kernel_size for each convolutional layer
        conv1_kernel_size = 2
        conv2_kernel_size = 2

        #Defining kernel_size for pooling after each convolutional layer
        self.pool1_kernel_size = 1
        self.pool2_kernel_size = 1

        #Defining the amount of measurements taken by the 28 EEG channels
        nb_measurements = 50

        #Defining number of input and output channels for the convolutional layers
        conv1_nb_in_channels = 28
        conv1_nb_out_channels = 10
        conv2_nb_out_channels = 5

        #Determining the input size of the first linear layer as a function of the previous operations
        self.linear1_in_size = conv2_nb_out_channels * floor((floor((nb_measurements-conv1_kernel_size+1)/self.pool1_kernel_size)-conv2_kernel_size+1)/self.pool2_kernel_size)

        #Defening each convolutional and linear layer with its respective input and output sizes
        self.conv1 = nn.Conv1d(conv1_nb_in_channels,conv1_nb_out_channels,kernel_size=conv1_kernel_size)
        self.conv2 = nn.Conv1d(conv1_nb_out_channels,conv2_nb_out_channels,kernel_size=conv2_kernel_size)
        self.fc1 = nn.Linear(self.linear1_in_size,10) #does the linear layer correctly handle batches?
        self.fc2 = nn.Linear(10,2)

    #Defining the different operations on the data in the right order
    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.conv1(x), kernel_size = self.pool1_kernel_size))
        x = F.tanh(F.max_pool1d(self.conv2(x), kernel_size = self.pool2_kernel_size))
        x = F.tanh(self.fc1(x.view(-1,self.linear1_in_size)))
        x = self.fc2(x)
        return x

#Choosing the size of the validation datasets and the lambda value used for L1 penalization
validate_size = 50
lambda_1 = 0.0001

#Loading dataset
train_input_base, train_target_base = bci.load(root = './data_bci')
test_input, test_target = bci.load(root = './data_bci', train = False)

#Creation of cross validation datasets with size 'validate_size'. 
train_input, train_target, validate_input, validate_target = cross_val_datasets(train_input_base, train_target_base, validate_size)

#Making each dataset Variable for allowing autograd functionality. 
validate_input, validate_target, train_target, train_input = Variable(validate_input), Variable(validate_target), Variable(train_target), Variable(train_input)
test_target, test_input = Variable(test_target), Variable(test_input)

#For Loop iterates through each combination of train and validation datasets created by the cross_val_datasets function
for i in range(train_input.size(0)):
    model, criterion = Net(), nn.CrossEntropyLoss()    

    #Normalizing data
    mu, std = train_input.data.mean(), train_input[i].data.std()
    train_input[i].data.sub_(mu).div_(std)

    #Instantiating learning rate, number of epoch and mini batch size
    lr, nb_epochs, batch_size = 1e-1, 100, 10

    #Selecting Stochastic Gradient Descent as model optimizer
    optimizer = optim.SGD(model.parameters(), lr = lr)

    #Number of epochs defines the amount of the times the entire dataset in put through the model in order to optimize it
    for k in range(nb_epochs):
        sum_loss = 0
        #Allows to split dataset up in smaller batches of size 'batch_size'
        for b in range(0, train_input[i].size(0) - train_input[i].size(0)%batch_size, batch_size): 
            #Putting one batch of the training data in the model    
            output = model(train_input[i].narrow(0, b, batch_size))
            #Calculating loss by comparing the output of the model with the expected targets
            loss = criterion(output, train_target[i].narrow(0,b,batch_size))
            #Setting gradient tensors back to zero 
            model.zero_grad()
            #Calculating the gradient tensors
            loss.backward()
            #Updating the parameters of the model
            optimizer.step()
            #Implementing L1 penalization
            for p in model.parameters():
                p.data -= p.data.sign() * p.data.abs().clamp(max = lambda_1)
            sum_loss = sum_loss + loss.data[0]

        #print(k,sum_loss)
        #print(k," Train Accuracy:",100*(1-compute_nb_errors(model,train_input,train_target,4)/len(train_input)))
        #print(k," Validate Accuracy:",100*(1-compute_nb_errors(model,validate_input,validate_target,4)/len(test_input)))
        #print(k," Test Accuracy:",100*(1-compute_nb_errors(model,test_input,test_target,4)/len(test_input)))
    #print(i,sum_loss)
    print(i," Train Accuracy:",100*(1-compute_nb_errors(model,train_input[i],train_target[i],batch_size)/len(train_input[i])))
    print(i," Validate Accuracy:",100*(1-compute_nb_errors(model,validate_input[i],validate_target[i],batch_size)/len(validate_input[i])))
    #print(i," Test Accuracy:",100*(1-compute_nb_errors(model,test_input,test_target,batch_size)/len(test_input)))
    print("-------------------------------------------------------------")