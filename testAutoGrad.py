import torch
from torch import autograd
from torch import LongTensor
from torch.autograd import Variable
from torch import FloatTensor as Tensor
import math


class ReLUAG(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		print(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		(input,) = ctx.saved_tensors
		print(input)
		return grad_output.mul((input > 0).double())
	
reluAG = ReLUAG.apply

class TanhAG(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return input.tanh()

	@staticmethod
	def backward(ctx, grad_output):
		(input,) = ctx.saved_tensors
		return grad_output.mul(1-input.tanh().pow(2))
	
tanhAG = TanhAG.apply



from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.

inputs = ((Variable(torch.randn(20,20).double(), requires_grad=True),))
#print(input)
test1 = gradcheck(TanhAG.apply, inputs, eps=1e-6, atol=1e-4)
print(test1)
 

inputs2 = (Variable(torch.randn(2,2).double(), requires_grad=True),)
test2 = gradcheck(ReLUAG.apply, inputs2, eps=1e-6, atol=1e-4)
print(test2)