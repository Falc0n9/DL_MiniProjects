from torch import LongTensor
from torch import FloatTensor as Tensor
import math


class Parameter(object):
    """
    Helper class to store parameter data with its gradient
    """

    def __init__(self, *args):
        """
        Initialize a parameter.
        :param args: tensor dimensions
        """
        self.data = Tensor(*args)
        self.grad = Tensor(*args)
        self.zero_grad()

    def zero_grad(self):
        """
        Zeros the gradient of a parameter
        """
        self.grad.zero_()


class Module(object):
    """
    Base class for a neural network module.
    """

    def forward(self, x):
        """
        Defines the computation performed at a forward call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Defines the computation performed at backpropagation.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def param(self):
        """
        Returns a list containing the parameters of a module.
        Should be overridden by all subclasses with parameters.
        """
        return []

    def zero_grad(self):
        """
        Zeros the gradient of all the module's parameters
        """
        for parameter in self.param():
            parameter.zero_grad()


class Tanh(Module):
    """
    Class to implement the tanh activation function.
    """

    def __init__(self):
        self.y = None

    def forward(self, x):
        """
        Implements tanh.
        Stores the result of the tanh operation for the backward pass.
        """
        self.y = x.tanh()
        return self.y

    def backward(self, grad_output):
        """
        Backpropagation for tanh.
        :param grad_output: the gradient of the loss with respect to the module's output
        :return: the gradient of the loss with respect to the module's input
        """
        return grad_output.mul(1 - self.y.pow(2))


class ReLU(Module):
    """
    Cass to implement the ReLU activation function.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Implements ReLU.
        Stores the input for the backward pass.
        """
        self.x = x
        return x.clamp(min=0)

    def backward(self, grad_output):
        """
        Backpropagation for ReLU.
        :param grad_output: the gradient of the loss with respect to the module's output
        :return: the gradient of the loss with respect to the module's input
        """
        return grad_output.mul((self.x > 0).float())


class Linear(Module):
    """
    Class to implement a linear layer.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize a linear layer.
        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: whether to include a bias term
        """

        self.weight = Parameter(out_features, in_features)
        if bias:
            self.bias = Parameter(out_features)
        else:
            self.bias = None

        self.x = None
        self.init_parameters()

    def init_parameters(self):
        """
        Xavier's parameter initialization rule.
        """
        stdv = 1. / math.sqrt(self.weight.data.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Implements a Linear pass.
        Stores the input for the backward pass.
        """
        self.x = x
        if self.bias is not None:
            return x.mm(self.weight.data.t()) + self.bias.data
        else:
            return x.mm(self.weight.data.t())

    def backward(self, grad_output):
        """
        Backpropagation for Linear.
        Accumulates the gradient in the weight and bias parameters.
        :param grad_output: the gradient of the loss with respect to the module's output
        :return: the gradient of the loss with respect to the module's input
        """

        grad_input = grad_output.mm(self.weight.data)
        grad_weight = grad_output.t().mm(self.x)
        grad_bias = grad_output.sum(0).squeeze(0)

        self.weight.grad += grad_weight
        if self.bias is not None:
            self.bias.grad += grad_bias

        return grad_input

    def param(self):
        """
        :return: weight and bias parameters of a linear layer
        """
        result = [self.weight]
        if self.bias is not None:
            result += [self.bias]
        return result


class Sequential(Module):
    """
    Class to sequentially connect layers and acivation functions.
    """

    def __init__(self, *args):
        """
        :param args: variable amount of submodules (layers and activation functions).
        """
        self.modules = args

    def forward(self, x):
        """
        Implements a sequential pass.
        Calls the forward pass on each submodule with the output of the previous submodule as input.
        """
        result = x
        for module in self.modules:
            result = module.forward(result)
        return result

    def backward(self, grad_output):
        """
        Implements sequential backpropagation.
        Cals the backward pass on each submodule with the output of the next submodule as input.
        :param grad_output: the gradient of the loss with respect to the module's output
        :return: the gradient of the loss with respect to the module's input
        """
        grad_input = grad_output
        for module in reversed(self.modules):
            grad_input = module.backward(grad_input)
        return grad_input

    def param(self):
        """
        :return: parameters of all submodules
        """
        result = []
        for module in self.modules:
            result += module.param()
        return result


def loss_mse(x, y):
    """
    Computes the mean-square error loss.
    :param x: input data
    :param y: input target
    :return: Mean-square error of data wrt target, gradient of the MSE wrt data
    """
    error = (x - y)
    loss = error.pow(2).sum()
    loss_grad = 2 * error
    return loss, loss_grad
