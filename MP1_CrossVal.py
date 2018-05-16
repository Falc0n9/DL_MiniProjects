import dlc_bci as bci
from math import ceil, floor
from torch import cuda, nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets
from dlc_practical_prologue import convert_to_one_hot_labels
from torch import LongTensor
from cross_val import cross_val_datasets

#Adapt Learning Rate Depending on Batch Size
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        conv1_kernel_size = 2
        conv2_kernel_size = 2

        self.pool1_kernel_size = 1
        self.pool2_kernel_size = 1

        nb_measurements = 50

        conv1_nb_in_channels = 28
        conv1_nb_out_channels = 10
        conv2_nb_out_channels = 5

        self.linear1_in_size = conv2_nb_out_channels * floor((floor((nb_measurements-conv1_kernel_size+1)/self.pool1_kernel_size)-conv2_kernel_size+1)/self.pool2_kernel_size)

        self.conv1 = nn.Conv1d(conv1_nb_in_channels,conv1_nb_out_channels,kernel_size=conv1_kernel_size)
        self.conv2 = nn.Conv1d(conv1_nb_out_channels,conv2_nb_out_channels,kernel_size=conv2_kernel_size)
        self.fc1 = nn.Linear(self.linear1_in_size,10) #does the linear layer correctly handle batches?
        self.fc2 = nn.Linear(10,2)

    def forward(self, x):
        x = F.tanh(F.max_pool1d(self.conv1(x), kernel_size = self.pool1_kernel_size))
        x = F.tanh(F.max_pool1d(self.conv2(x), kernel_size = self.pool2_kernel_size))
        x = F.tanh(self.fc1(x.view(-1,self.linear1_in_size)))
        x = self.fc2(x)
        return x


validate_size = 30
lambda_1 = 0.0001

train_input_base, train_target_base = bci.load(root = './data_bci')
test_input, test_target = bci.load(root = './data_bci', train = False)

train_input, train_target, validate_input, validate_target = cross_val_datasets(train_input_base, train_target_base, validate_size)
validate_input, validate_target, train_target, train_input = Variable(validate_input), Variable(validate_target), Variable(train_target), Variable(train_input)

test_target = Variable(test_target)
test_input = Variable(test_input)

for i in range(train_input.size(0)):
    model, criterion = Net(), nn.CrossEntropyLoss()    

    mu, std = train_input.data.mean(), train_input[i].data.std()
    train_input[i].data.sub_(mu).div_(std)

    lr, nb_epochs, batch_size = 1e-1, 100, 10

    optimizer = optim.SGD(model.parameters(), lr = lr)

    def compute_nb_errors(model, input, target, mini_batch_size):

        nb_errors = 0

        for b in range(0, input.size(0) - input.size(0)%mini_batch_size, mini_batch_size):
            output = model(input.narrow(0, b, mini_batch_size))
            _, predicted_classes = output.data.max(1)
            for k in range(0, mini_batch_size):
                if target.data[b + k] != predicted_classes[k]:
                    nb_errors = nb_errors + 1

        return nb_errors

    for k in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input[i].size(0) - train_input[i].size(0)%batch_size, batch_size):     
            output = model(train_input[i].narrow(0, b, batch_size))
            loss = criterion(output, train_target[i].narrow(0,b,batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
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
