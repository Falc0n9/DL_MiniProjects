from torch import Tensor
from mini_project_2 import *
from math import pi
import matplotlib.pyplot as plt
from helperfunctions import plot_data,convert_to_one_hot_labels, compute_nb_errors

train_data = Tensor(10000, 2).uniform_()
train_target = (train_data.pow(2).sum(1) < 1 / (2 * pi)).long()
train_target_one_hot = convert_to_one_hot_labels(train_data, train_target)

seq = Sequential(Linear(2, 25),
                 ReLU(),
                 Linear(25, 25),
                 ReLU(),
                 Linear(25, 25),
                 ReLU(),
                 Linear(25, 2))

lr, nb_epochs, batch_size = 0.01, 40, 10

loss_list = []
for e in range(0, nb_epochs):
    sum_loss = 0
    for b in range(0, train_data.size(0), batch_size):

        data = train_data.narrow(0, b, batch_size)
        target = train_target_one_hot.narrow(0, b, batch_size)

        seq.zero_grad()
        loss, loss_grad = loss_mse(seq.forward(data), target)
        seq.backward(loss_grad)
        sum_loss = sum_loss + loss
        for p in seq.param():
            p.data.sub_(lr * p.grad)
    loss_list += [sum_loss]

plt.plot(loss_list)
plt.show()

output = seq.forward(train_data)
_, predicted_classes = output.max(1)
plot_data(train_data, predicted_classes)

nb_errors = compute_nb_errors(seq.forward, train_data, train_target, batch_size)
nb_data = train_target.size(0)
accuracy = 100-(nb_errors/nb_data)*100
print('train error {:d}/{:d} = {:0.2f}% accuracy'.format(nb_errors,nb_data,accuracy))
