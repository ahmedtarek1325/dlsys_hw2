"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        
        for parameter in self.params: 
            grad = self.u.get(parameter,0)*self.momentum + \
                    (1- self.momentum)* (parameter.grad + self.weight_decay * parameter.data)
            
            grad = ndl.Tensor(grad,dtype=parameter.dtype)
            self.u[parameter] = grad
            
            parameter.data = parameter- self.lr* grad
                    
        return 
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t+=1
        for param in self.params:

            grad = param.grad.data + self.weight_decay * param.data
            grad = ndl.Tensor(grad,dtype=param.dtype)

            u_t_1 = self.beta1*self.m.get(param,0) +(1-self.beta1) * grad
            self.m[param] = u_t_1 
            
            v_t_1 = self.beta2*self.v.get(param,0) +(1-self.beta2) * (grad**2)
            self.v[param] = v_t_1
            
            u_t_1 /= (1-self.beta1**self.t)
            v_t_1 /= (1-self.beta2**self.t)

            param.data = param.data - self.lr* u_t_1 / (v_t_1**0.5 + self.eps)
                    
        return 
        ### END YOUR SOLUTION
