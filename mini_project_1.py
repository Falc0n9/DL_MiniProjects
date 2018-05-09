import dlc_bci as bci
from math import ceil
from torch import cuda, nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets
from dlc_practical_prologue import convert_to_one_hot_labels
from torch import LongTensor










class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        conv1_kernel_size = 3
        conv2_kernel_size = 3

        self.pool1_kernel_size = 3
        self.pool2_kernel_size = 2

        nb_measurements = 50

        conv1_nb_in_channels = 28
        conv1_nb_out_channels = 32
        conv2_nb_out_channels = 64

        self.linear1_in_size = conv2_nb_out_channels * ceil((ceil((nb_measurements-conv1_kernel_size+1)/self.pool1_kernel_size)-conv2_kernel_size+1)/self.pool2_kernel_size)

        self.conv1 = nn.Conv1d(conv1_nb_in_channels,conv1_nb_out_channels,kernel_size=conv1_kernel_size)
        self.conv2 = nn.Conv1d(conv1_nb_out_channels,conv2_nb_out_channels,kernel_size=conv2_kernel_size)
        self.fc1 = nn.Linear(self.linear1_in_size,200)
        self.fc2 = nn.Linear(200,2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size = self.pool1_kernel_size))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size = self.pool2_kernel_size))
        x = F.relu(self.fc1(x.view(-1,self.linear1_in_size)))
        x = self.fc2(x)
        return x


train_input, train_target = bci.load(root = './data_bci')
test_input, test_target = bci.load(root = './data_bci', train = False)
train_target = Variable(train_target)
train_input = Variable(train_input.float())
test_input = Variable(test_input.float())
test_target = Variable(test_target)


model, criterion = Net(), nn.CrossEntropyLoss()

mu, std = train_input.data.mean(), train_input.data.std()
train_input.data.sub_(mu).div_(std)

lr, nb_epochs, batch_size = 1e-2, 100, 4

optimizer = optim.SGD(model.parameters(), lr = lr)


for k in range(nb_epochs):
    sum_loss = 0
    for b in range(0, train_input.size(0), batch_size):
        output = model(train_input.narrow(0, b, batch_size))
        loss = criterion(output, train_target.narrow(0,b,batch_size))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss = sum_loss + loss.data[0]

    print(k,sum_loss)


def compute_nb_errors(model, input, target, mini_batch_size):

    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(0, mini_batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

print(test_input.size())
print(compute_nb_errors(model,test_input,test_target,4))


