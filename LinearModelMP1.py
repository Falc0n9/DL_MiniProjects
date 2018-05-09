# This is distributed under BSD 3-Clause license

import torch
import torch.nn as nn
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
from dlc_bci import load

'''
LOADING DATASET
'''

inputs, labels = load('~ Documents/Studies/EPFL/Semester2/DeepLearning/MiniProject1/data_bci/',train = True, download = False, one_khz = False)
inputs = Variable(inputs.view(-1, 28*50))
labels = Variable(labels)
print(labels)
'''
#CREATE MODEL CLASS
'''

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

'''
#INSTANTIATE MODEL CLASS
'''
input_dim = 28*50
output_dim = 1

epochs = 100

model = LinearRegressionModel(input_dim, output_dim)

'''
#INSTANTIATE LOSS CLASS
'''

criterion = nn.CrossEntropyLoss()

'''
#INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
#TRAIN THE MODEL
'''

for epoch in range(epochs):
    epoch += 1
   
    # Clear gradients w.r.t. parameters
    optimizer.zero_grad() 
    
    # Forward to get output
    outputs = model(inputs)
    
    # Calculate Loss
    loss = criterion(outputs, labels)
    
    # Getting gradients w.r.t. parameters
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    # Logging
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))