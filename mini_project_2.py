from torch import LongTensor
from torch import FloatTensor as Tensor
import math

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        raise NotImplementedError

class Function(Module): # a function is a module without params

    @staticmethod
    def forward(x):
        r"""
        This function is to be overridden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def backward(gradwrtoutput):
        r"""
        This function is to be overridden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def param():
        return []

class Tanh(Function):

    @staticmethod
    def forward(x):
        result = x.clone()
        return result.tanh() #inplace or not?

    @staticmethod
    def backward(x):
        result = x.clone()
        return 1-result.tanh().pow(2)

    #param(self) is implemented in Functional super class


class ReLU(Function):

    @staticmethod
    def forward(x):
        result = x.clone()
        return result.clamp(min=0)

    @staticmethod
    def backward(x):
        return x > 0


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):

        self.weight = Tensor(out_features,in_features)
        self.weight_grad = Tensor(self.weight.size())

        if bias:
            self.bias = Tensor(out_features)
            self.bias_grad = Tensor(self.bias.size())
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def zero_grad(self):
        self.weight_grad.zero_()
        self.bias_grad.zero_()

    def forward(self, ctx, x):
        ctx.x = x

        if self.bias is not None:
            return self.weight.mm(x) + self.bias
        else:
            return self.weight.mm(x)

    def backward(self, ctx, grad_output):

        grad_input = grad_output.mm(self.weight)
        grad_weight = grad_output.t().mm(ctx.x)
        grad_bias = grad_output.sum(0).squeeze(0)

        self.weight_grad += grad_weight
        self.bias_grad += grad_bias

        return grad_input



