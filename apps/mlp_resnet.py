import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # I don't think that adding biases in those linear model is useful
    # since we add a norm after the linear model. But am following the
    # assignment architecture to pass the mugrade tests
    f = nn.Sequential(nn.Linear(dim,hidden_dim),
                           norm(hidden_dim),
                           nn.ReLU(),
                           nn.Dropout(drop_prob), 
                           nn.Linear(hidden_dim,dim),
                           norm(dim))

    model = nn.Sequential(nn.Residual(f),
                           nn.ReLU())
    return model
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    #residual_block = ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob)
    #Rblocks = nn.Sequential(*[residual_block]*num_blocks)
    ''' final_model = nn.Sequential(nn.Linear(dim,hidden_dim),
                                nn.ReLU(),
                                nn.Sequential(*[ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob)]*num_blocks),
                                nn.Linear(hidden_dim,num_classes))'''
    
    final_model = nn.Sequential(nn.Linear(dim,hidden_dim),
                                nn.ReLU(),
                                repeate_Rblock(hidden_dim,norm,drop_prob,num_blocks),
                                nn.Linear(hidden_dim,num_classes))
    return final_model
    ### END YOUR SOLUTION
def repeate_Rblock(hidden_dim,norm,drop_prob,num_blocks):
    '''
        Although I can replace this with the a simple 
        nn.SSequential([ResidualBlock]*num_blocks)
        and would have the ssame exact structure. The Intialization would be different. 
        to accomodat this I built this simple function to pass mugrade; Mugrade is the
        online grader for this assignment

    '''
    sequenes = []
    for i in range(num_blocks):
        residual_block = ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob)
        sequenes.append(residual_block)
    return nn.Sequential(*sequenes)





def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train()
    if opt is None: 
        model.eval()
    
    accuracy = avg_loss = 0 
    loss_func = nn.SoftmaxLoss()
    
    for _,(Xs,ys) in enumerate(dataloader):
        predicted = model(Xs)
        loss = loss_func(predicted,ys)
        
        avg_loss+= loss.numpy() * ys.shape[0]
        accuracy += np.sum(predicted.numpy().argmax(axis= 1) == ys.numpy()) 

        if opt: 
            loss.backward()
            opt.step()
            opt.reset_grad()
            
    avg_error_rate = 1- accuracy/len(dataloader.dataset)
    return avg_error_rate,avg_loss/len(dataloader.dataset)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    
    training_dataloader = ndl.data.DataLoader(train_dataset, batch_size,True)
    testing_dataloader = ndl.data.DataLoader(test_dataset, batch_size)

    # 784 is 28*28 which is the imagge size in MNIST dataset
    model = MLPResNet(784, hidden_dim)

    optimizer = optimizer(model.parameters(),lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        err_train,loss_train = epoch(training_dataloader,model,optimizer)
        err_test,loss_test = epoch(training_dataloader,model)
        print(f" Train error: {err_train} || train loss : {loss_train}||",end="")
        print(f"test err: {err_test} || test loss {loss_test}")
    
    return (err_train,loss_train,err_test,loss_test)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
