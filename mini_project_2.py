from torch import LongTensor
from torch import FloatTensor as Tensor
import math

#TODO: maybe add support for minibatches

class Module(object):

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def param(self):
        raise NotImplementedError

    def has_param(self):
        return True

class Function(Module): # a function is a module without params

    @staticmethod
    def forward(x):
        r"""
        This function is to be overridden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def backward(grad_output):
        r"""fc
        This function is to be overridden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def param():
        return []

    @staticmethod
    def has_param(self):
        return False

class Tanh(Function):

    @staticmethod
    def forward(x):
        result = x.clone()
        return result.tanh() #inplace or not?

    @staticmethod
    def backward(grad_output):
        result = grad_output.clone()
        return 1-result.tanh().pow(2) # klopt niet

    #param(self) is implemented in Functional super class


class ReLU(Function):

    #ctx

    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        result = x.clone()
        return result.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        result = grad_output.mul(ctx.x>0) #double check for x = 0
        return result


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):

        self.weight = Tensor(out_features,in_features)
        self.weight_grad = Tensor(self.weight.size()).zero_()

        if bias:
            self.bias = Tensor(out_features)
            self.bias_grad = Tensor(self.bias.size()).zero_()
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

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

    def param(self):
        return [(self.weight,self.weight_grad),(self.bias,self.bias_grad)]

class Sequential(Module):


    def __init__(self, *args):
        self.modules = args

    def forward(self, x):
        result = x.copy()
        for module in self.modules: # traverse in sequential direction
            result = module.forward(result)

        return result


    def backward(self, grad_output):

        grad_input = grad_output.copy()
        for module in self.modules:
            grad_input = module.backward(grad_input) #klopt misschien niet

        return grad_input

    def zero_grad(self):

        for module in self.modules:
            if module.has_param():
                module.zero_grad()

    def param(self):

        result = []

        for module in self.modules:
            if module.has_param():
                result += module.param()

        return result





