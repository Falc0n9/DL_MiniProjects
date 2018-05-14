# This is distributed under BSD 3-Clause license

import torch
import torch.nn as nn
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
from dlc_bci import load

'''
STEP 1: LOADING DATASET
'''

inputs, labels = load('./data_bci/',train = True, download = False, one_khz = False)
inputs = Variable(inputs.view(-1, 28*50))
labels = Variable(labels)

'''
STEP 2:
'''

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(inputs) / batch_size)
num_epochs = int(num_epochs)

'''
STEP 3: CREATE MODEL CLASS
'''
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28*50
output_dim = 1

model = LogisticRegressionModel(input_dim, output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()


'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''

for epoch in range(num_epochs):
    
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        outputs = model(inputs)
       
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        
        # Logging
        print('epoch {}, loss {}'.format(epoch, loss.data[0]))