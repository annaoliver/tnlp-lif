#Sun Y, Zhao F, Zhao Z, et al. Multi-compartment neuron and population encoding powered spiking neural network for deep distributional reinforcement learning[J]. Neural Networks, 2025, 182: 106898.
#https://github.com/BrainCog-X/Brain-Cog/
import math

import torch
from torch import nn
from torch.nn import functional as F


def heaviside(x):
    return (x >= 0.).to(x.dtype)


class SurrogateFunctionBase(nn.Module):
    """
    The base class for Surrogate Functions

    :param alpha: Provides parameters for surrogate functions that can adjust the shape of the function.

    :param requires_grad: Whether the parameter `alpha` needs to be computed; defaults to `False`.
    """

    def __init__(self, alpha, requires_grad=True):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        """
        :param x: Input to the membrane potential

        :param alpha: Variable controlling the shape of the surrogate gradient, can be ``NoneType`

        :return: Spike value after excitation, takes the value ``[0, 1]`
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: Membrane potential input

        :return: Spike after excitation
        """
        return self.act_fun(x, self.alpha)


'''
    sigmoid surrogate function.
'''


class sigmoid(torch.autograd.Function):
    """
    Using sigmoid as a surrogate gradient function

    The corresponding original function is:

    .. math::

    g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}

    The function for backpropagation is:

    .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = grad_output * s_x * (1 - s_x) * ctx.alpha
        return grad_x, None


class SigmoidGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid.apply(x, alpha)


'''
    atan surrogate function.
'''


class atan(torch.autograd.Function):
    """
    Using Atan as the surrogate gradient function, the corresponding primitive function is:

    .. math::
    g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}

    The function for backpropagation is:

    .. math::
    g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}

    """

    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        return inputs.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None

        shared_c = grad_output / \
                   (1 + (ctx.saved_tensors[1] * math.pi /
                         2 * ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.saved_tensors[1] / 2 * shared_c
        if ctx.needs_input_grad[1]:
            grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()

        return grad_x, grad_alpha


class AtanGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return atan.apply(x, alpha)


'''
    gate surrogate fucntion. 
'''


class gate(torch.autograd.Function):
    """
    Using `gate` as the surrogate gradient function, the corresponding primitive function is:

    .. math::

    g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)

    The function for backpropagation is:

    .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            grad_x = torch.where(x.abs() < 1. / alpha, torch.ones_like(x), torch.zeros_like(x))
            ctx.save_for_backward(grad_x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class GateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return gate.apply(x, alpha)


'''
    gatquadratic_gate surrogate function.
'''


class quadratic_gate(torch.autograd.Function):
    """
    Using quadratic_gate as a surrogate gradient function

    The corresponding original function is:

    .. math::
        g(x) =
        \\begin{cases}
        0, & x < -\\frac{1}{\\alpha} \\\\
        -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
        1, & x > \\frac{1}{\\alpha} \\\\
        \\end{cases}

    The function for backpropagation is:

    .. math::
        g'(x) =
        \\begin{cases}
        0, & |x| > \\frac{1}{\\alpha} \\\\
        -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
        \\end{cases}

    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            mask_zero = (x.abs() > 1 / alpha)
            grad_x = -alpha * alpha * x.abs() + alpha
            grad_x.masked_fill_(mask_zero, 0)
            ctx.save_for_backward(grad_x)
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class QGateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return quadratic_gate.apply(x, alpha)


class relu_like(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_alpha = None, None
        x, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * x.gt(0.).float() * alpha
        if ctx.needs_input_grad[1]:
            grad_alpha = (grad_output * F.relu(x)).sum()
        return grad_x, grad_alpha

class RoundGrad(nn.Module):
    def __init__(self, **kwargs):
        super(RoundGrad, self).__init__()
        self.act = nn.Hardtanh(-.5, 4.5)

    def forward(self, x):
        x = self.act(x)
        return x.ceil() + x - x.detach()

class ReLUGrad(SurrogateFunctionBase):
    """
    ReLU is used as an alternative to the gradient function, primarily for testing ANNs with the same structure.
    """

    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return relu_like.apply(x, alpha)


'''
    Straight-Through (ST) Estimator
'''


class straight_through_estimator(torch.autograd.Function):
    """
    Use the pass-through estimator as a surrogate gradient function
    http://arxiv.org/abs/1308.3432
    """

    @staticmethod
    def forward(ctx, inputs):
        outputs = heaviside(inputs)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x


class stdp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        outputs = inputs.gt(0.).float()
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        return inputs * grad_output


class STDPGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return stdp.apply(x)





class backeigate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()


class BackEIGateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return backeigate.apply(x)

class ei(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()


class EIGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return ei.apply(x)


