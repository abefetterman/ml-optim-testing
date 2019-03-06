import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# What we want to do is have the update be
# sgd_update = - mu * grad
# adam_update = - mu * grad / (grad**2)
# new update = - mu * grad * (beta2 + beta1) / (grad**2 * beta1 + avg(grad**2) * beta2)
# where beta1 is gradually fading in (basically the reverse of the adam 
# correction for early times) and beta2 is some factor, probably 2 ish
# beta2 -> infinity is sgd (w wonky learning rate), beta2->0 is adam.
# Goal is to surpress high eigenvalue eigenvectors while leaving the rest alone

class SmoothAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 eta=2.0, weight_decay=0, nesterov=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, eta=eta,
                        weight_decay=weight_decay, 
                        nesterov=nesterov)
        super(SmoothAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                # beta1_hat = beta1 #min(beta1,1.0-1.0/state['step'])
                beta2_hat = min(beta2,1.0-1.0/state['step'])
                fade_in = min(1.0,(1-beta2)*(1+state['step']))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2_hat).addcmul_(1 - beta2_hat, grad, grad)
                denom_smooth = torch.mean(exp_avg_sq) * group['eta'] + group['eps']*group['eps']

                denom = exp_avg_sq.mul(fade_in).add_(denom_smooth).sqrt()
                num_coeff = group['eta'] + fade_in + group['eps']

                wd = group['weight_decay']*group['lr']

                p.data.add_(-wd, p.data)

                if nesterov:
                    p.data.addcdiv_(-group['lr']*beta1*num_coeff, exp_avg, denom)
                    p.data.addcdiv_(-group['lr']*(1-beta1)*num_coeff, grad, denom)
                else:
                    p.data.addcdiv_(-group['lr']*num_coeff, exp_avg, denom)

        return loss
