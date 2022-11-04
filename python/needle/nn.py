"""The module.
"""
from turtle import shape
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

from python import needle


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight= Parameter(init.kaiming_uniform(self.in_features,self.out_features))
        self.bias = init.kaiming_uniform(self.out_features,1) if bias else []
        if self.bias: 
            self.bias = Parameter(self.bias.transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
            # if u have not broadcasted the self.bias, then its gradient
            # will not be calculated in the gradient computations 
            # resultsing in very small but hurtful numerical error 
            return  X@self.weight + self.bias.broadcast_to((X.shape[0],self.out_features))
        else: 
            return X@self.weight
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        X= ops.Reshape((X.shape[0],-1))(X)
        return X
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.ReLU()(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for layer in self.modules: 
            x=layer(x)
        return x 
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_label = init.one_hot(logits.shape[1],y)
        zi = logits*y_label
        zi = ops.summation(zi,axes=1)

        softmax = ops.LogSumExp(axes=1)(logits)
        loss= softmax-zi
        loss = ops.summation(loss)/loss.shape[0]

        return loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # intialize weight and bias to ones and zeros 
        self.weight = Parameter(init.ones(self.dim))
        self.bias  = Parameter(init.zeros(self.dim))
        self.running_mean = init.zeros(self.dim)
        self.running_var  = init.ones(self.dim)
        self.training = True
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training: 
            return self.forward_trainning(x)
        else: 
            return self.forward_testing(x)
        ### END YOUR SOLUTION
    def eval(self)-> None:
        self.training = False
    
    def forward_trainning(self,x:Tensor)->Tensor:
        mean_ = self.calculate_mean(x)
        mean= self.broadcasting_type2(mean_,x)

        var_ = self.calculate_var(x,mean)
        var= self.broadcasting_type2(var_,x)
        # broadcast weight and bias so that they will have the 
        # same shape of x
        y = self.broadcasting_type2(self.weight,x) * \
            (x-mean)/((var+self.eps)**0.5) + self.broadcasting_type2(self.bias,x)
        
        # now we want to update the remmainning 
        # mean/std 
        self.running_mean = mean_*self.momentum + \
            (1-self.momentum)*self.running_mean
        self.running_var = var_*self.momentum + \
            (1-self.momentum)*self.running_var
        self.running_mean = self.running_mean.detach()
        self.running_var= self.running_var.detach()
        return y 
    
    def forward_testing(self,x:Tensor)->Tensor:
        mean= self.broadcasting_type2(self.running_mean,x)
        var= self.broadcasting_type2(self.running_var,x)
        normalized_X= (x-mean)/((var+self.eps)**0.5)

        y= self.broadcasting_type2(self.weight,x) *normalized_X \
            + self.broadcasting_type2(self.bias,x) 
        return y 


    def calculate_mean(self,x:Tensor)-> Tensor:
        '''
        calculate the mean of the batch an broadcast it
        to the shape of the passed tensor
        INPUT 
        x: Tensor with dim (batch_size,feature_size)
        OUTPUT
        mean: Tensor with dim (feature_size,)
        '''
        # we get the mean over the batch itself
        mean= ops.divide_scalar(ops.summation(x,axes = 0), x.shape[0]) 
        return mean
    def calculate_var(self,x:Tensor,mean:Tensor)-> Tensor:
        '''
        calculate the var of the batch an broadcast it
        to the shape of the passed tensors
        INPUT 
        x: Tensor with dim (batch_size,feature_size)
        mean: Tensor with dim (batch_size,feature_size)
        OUTPUT
        var: Tensor with dim (feature_size,)
        '''
        # we get the mean over the batch itself
        var = ops.summation(ops.power_scalar((x-mean),2),axes=0) / x.shape[0]
        return var

    def broadcasting_type2(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the weights 
        and bias which basically had 
        (batch,features)-->(features,)
        '''
        x1 = x1.reshape((1,-1))
        x1 = x1.broadcast_to(x.shape)
        return x1
    def broadcasting_type1(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the mean 
        and var which basically had 
        (batch,features)-->(features,)
        '''
        x1 = x1.reshape((1,-1))
        x1 = x1.broadcast_to(x.shape)
        return x1

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # intialize weight and bias to ones and zeros 
        self.weight = Parameter(init.ones(self.dim))
        self.bias  = Parameter(init.zeros(self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # a funciton to comupte the mean 
        # a function to compute the varience 
        # then appl the equation provided in the notebook 
        # is it Ewise subtraction ? 
        # 
        mean = self.mean(x)
        var = self.variance(x,mean)
        
        mean = self.broadcasting_type1(mean,x)
        var = self.broadcasting_type1(var,x)

        y= self.broadcasting_type2(self.weight,x) * ((x-mean)/((var+self.eps)**0.5)) + self.broadcasting_type2(self.bias,x)
        return y


        ### END YOUR SOLUTION
    def mean(self,x:Tensor)-> Tensor : 
        # calculate the mean of the tensor 
        return ops.divide_scalar(ops.summation(x,axes = 1), x.shape[1])
    
    def variance(self,x:Tensor,mean:Tensor)-> Tensor:
        # this function calculates the varience for us 
        mean = self.broadcasting_type1(mean,x)
        var = ops.summation(ops.power_scalar((x-mean),2),axes=1)
        var = ops.divide_scalar(var,x.shape[1])
        return var


    def broadcasting_type1(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the mean 
        and variance which basically had 
        (batch,features)-->(bath,)
        '''
        x1 = x1.reshape((-1,1))
        x1 = x1.broadcast_to(x.shape)
        return x1
    def broadcasting_type2(self,x1:Tensor,x:Tensor): 
        '''
        This function is used to broadcast the weights 
        and bias which basically had 
        (batch,features)-->(features,)
        '''
        x1 = x1.reshape((1,-1))
        x1 = x1.broadcast_to(x.shape)
        return x1


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p
        

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training: 
            sum_ = 1
            for i in x.shape : sum_*=i

            # getting array of zeros with prob = p and
            # reshaping it to've the same shape as x
            norm = init.randb(sum_,p = 1- self.p)
            norm = norm.reshape(x.shape)
            
            # zero some numbers and then rescale with (1-p)
            x = x*norm 
            x /=(1-self.p)
        return x 
        
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.EWiseAdd()(x,self.fn(x))
        ### END YOUR SOLUTION



