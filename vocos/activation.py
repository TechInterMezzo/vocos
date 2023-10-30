import torch
from torch import autograd

class APTx(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (1 + torch.tanh(input)) * input / 2

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            input = ctx.saved_tensors[0]
            grad_input = grad_output * (1 + torch.tanh(input) + input * sech(input) ** 2) / 2
        return grad_input
    
def sech(x):
  return 2 / (torch.exp(x) + torch.exp(-x))