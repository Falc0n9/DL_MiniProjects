from torch import LongTensor
from torch import FloatTensor as Tensor
import math


class Parameter(object):
    """
    A Parameter is a Tensor with an extra grad field.
    """

    def __init__(self, *args):
        self.data = Tensor(*args)
        self.grad = Tensor(*args)
        self.zero_grad()

    def zero_grad(self):
        self.grad.zero_()


class Module(object):

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        for parameter in self.param():
            parameter.zero_grad()


class Tanh(Module):

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = x.tanh()
        return self.y

    def backward(self, grad_output):
        return grad_output.mul(1 - self.y.pow(2))


class ReLU(Module):

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return x.clamp(min=0)

    def backward(self, grad_output):
        return grad_output.mul((self.x > 0).float())


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):

        self.weight = Parameter(out_features, in_features)
        if bias:
            self.bias = Parameter(out_features)
        else:
            self.bias = None

        self.x = None

        self.init_parameters()

    def init_parameters(self):  # TODO: check
        stdv = 1. / math.sqrt(self.weight.data.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        self.x = x
        if self.bias is not None:
            return x.mm(self.weight.data.t()) + self.bias.data
        else:
            return x.mm(self.weight.data.t())

    def backward(self, grad_output):

        grad_input = grad_output.mm(self.weight.data)
        grad_weight = grad_output.t().mm(self.x)
        grad_bias = grad_output.sum(0).squeeze(0)

        self.weight.grad += grad_weight
        self.bias.grad += grad_bias

        return grad_input

    def param(self):
        return [self.weight, self.bias]


class Sequential(Module):

    def __init__(self, *args):
        self.modules = args

    def forward(self, x):
        result = x
        for module in self.modules:
            result = module.forward(result)
        return result

    def backward(self, grad_output):
        grad_input = grad_output
        for module in reversed(self.modules):
            grad_input = module.backward(grad_input)
        return grad_input

    def param(self):
        result = []
        for module in self.modules:
            result += module.param()
        return result


def loss_mse(x, y):
    error = (x - y)

    loss = error.pow(2).sum()
    grad = 2 * error

    return loss, grad