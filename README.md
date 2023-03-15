# DL systems HW2
This is assignment is a part of [DL-systems course presented by CMU](https://dlsyscourse.org/). 



## What you will find in this repo


This HW is the second step for building our own DL library. In this HW, we used the defined operations in [HW1](https://github.com/ahmedtarek1325/dlsys_hw1) to create DL layers. The main idea here is we only need to define forward propagation in terms of the ops, and the backward propagation will be computed directly using the AD that will build a graph for it. Besides, we draft additional components such as initialization and optimization algorithms.


Navigating this repo you will find an implementation for the following: 
1.  Weight initialization algorithms [*init.py*](https://github.com/ahmedtarek1325/dlsys_hw2/tree/master/python/needle)
    * Xavier uniform
    * Xavier normal 
    * Kaiming uniform
    * Kaiming normal
2. The following fundamental nn Layers: [*nn.py*](https://github.com/ahmedtarek1325/dlsys_hw2/tree/master/python/needle)
    * Linear 
    * ReLU 
    * Sequential
    * LogSumExp
    * SoftmaxLoss
    * LayerNorm1d
    * BatchNorm1d
    * Flatten
    * Dropout
    * Residual

3. The following optimizers [*optim.py*](https://github.com/ahmedtarek1325/dlsys_hw2/tree/master/python/needle)
    * SGD
    * ADAM
4. Implement softmax loss using needle operations [*data folder*](https://github.com/ahmedtarek1325/dlsys_hw2/tree/master/data)
    * RandomFlipHorizontal
    * Random crop 
    * Dataaloader function
5. Traininig our first DL block using needle. [*apps folder*](https://github.com/ahmedtarek1325/dlsys_hw2/blob/master/apps/mlp_resnet.py)




## my AHA moments
1. **For implementing the softmax we use LogsumExp, but why try to use the logarthmic to impement the softmax?** 
    I've ansswered this in [HW0](https://github.com/ahmedtarek1325/dlsys_hw0). 


2. **Dropouts, normliztion and breaking the symmetry**

    In HW1, I've talked about how operations such as summation and reshaping do not alter the back-propagation stream. Well, using dropouts or normalization can mitigate this effect and introduce some randomization in the back-propagation backward stream. 

    In the case of the Dropout: 
        You are now deactivating some nodes in the forward propagation, which prevents it from getting updated in the back-propagation, which from my POV, does add more flexibility. 

    While in the case of the normalization: 
        you are more of weighting the different nodes depending on the normalizaiton process. 

    In both cases, you are not saying that the derivavtive of the ouput w.r.t inputs in "summation" or transpose or etc is no longer just ones, but more of a sophisticated that cannot be anticipated with  the time. 
    
    
3. **Are you trying to build your own computational graph using numpy and facing wiered errors on calculating the gradients?**

    On implementing many functions as layernorm, logSumExp, etc. 

    I often came across a situation where I have to make an operation like summation between 

    $A_{3\times5} + B_{1\times5}$
    
     and since in this assignment we are still using numpy as our backend. Hence, Numpy would deal with this summation operation normally      without any errors by broadcasting the matrix B.

    Thus I'll get perfectly nice results on the forward computation. But the back-propagation would simply break out saying:

    **Hello there, incompaiatble dimensions !! x@** 

    This is simply because the broadcasting that numpy has done by default was not recorded in the computational graph. 
    
    **Remember**, Our goal is to build needle operations that we will only use to draft new layers; having any operation done without being recorded in the computational graph would either result in an error -like the dim one above- or would result in wrong back propagation results.

4. **If you trying to re-implement the assignment and faced a problem passing the DL block in question 5**   *I guess this part may be more useful if u tried to implement the assignmnet and failed*

    This would be as a result of the random intialition for the weights. It was a problem that I and lots of colleagues enrolled in the online class has faced,  But what is the problem exactly? 
    
    well we have the residual block being called multiple timess inside the MLPResNet, so if you looked at my implementation, you'll find that at first I started initalizing the residual blocks multiple time like the following 
    `Rblocks = nn.Sequential(*[residual_block]*num_blocks)`

    then after reallocating the `Rblocks`, I start bulding the mlpResNet. This would give different intialization for the weights of the network from the ones that the fellows in CMU have assumed my colleagess and I would follow. Consequently, it did not pass the tests despite being a right implementation. To overcome that error, you have to build the network in the same order as allocated in the mlpResNet image. 

    
--- 

## Want to see more of the assignments ? 
### Click here to see the rest of the assignments and my take outts
1. [HW1](https://github.com/ahmedtarek1325/dlsys_hw1)
2. [HW2](https://github.com/ahmedtarek1325/dlsys_hw2)
3. [HW3](https://github.com/ahmedtarek1325/dlsys_hw3)
4. [HW4](https://github.com/ahmedtarek1325/dlsys_hw4)

## Refrences:  
1. [Lectures [6-10] to understand the theory](https://dlsyscourse.org/lectures/)
