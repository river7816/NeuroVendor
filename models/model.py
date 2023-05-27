# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:48:56 2023

@author: river
"""
import sys
import numpy as np
sys.path.append('../')

from utils.loss import cost
import torch.nn as nn
import torch.nn.functional as F 
import torch

import torch
import torch.nn as nn

# Define a single layer ReLU network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.b = nn.Parameter(torch.FloatTensor(1).uniform_(0, 1))

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        return x.sum() + self.b

# net2 = nn.Sequential(
#     nn.Linear(in_features=20, out_features=10),
#     nn.ReLU(),
#     nn.Linear(in_features=10, out_features=1)       
#               )



# net = Net(input_size=20, hidden_size=10)
# for name, param in net.named_parameters():
#     print(name, param, param.shape)
#%%        




class ModelCompiler(nn.Module):
    def __init__(self, model, loss_func=cost, perishable=True):
        '''
        

        Parameters
        ----------
        model : pytorch models
            eg: nn.Linear.
        loss_func : loss function, optional
             The default is cost.
        perishable :bool, optional
             perishabe or not. The default is False.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.u = torch.tensor(0.)
        self.model = model
        self.perishable = perishable
        self.loss_func = loss_func
    
    def forward(self, x, demand):
        '''
        

        Parameters
        ----------
        x : features
            shape: (1, N).
        demand : demand
            shape: (1, 1).

        Returns
        -------
        y : predict y
            if perishable, y = y_hat
            if unperishable, y = max(y_hat, u).
        loss : TYPE
            loss function
            h*[y-D]^{+} + b*[D-y]^{+}.

        '''
        y_hat = self.model(x)
        if self.perishable:
            y = y_hat # without contraints 
        else:
            y = torch.maximum(y_hat, self.u) # y = max(y_hat, u)
            self.u = (F.relu(y-demand)).data # update u .data means no grad
        
        loss = self.loss_func(y_hat, demand)
        cost = self.loss_func(y, demand)
        return y, loss, cost

def relu(x):
    return(np.maximum(0, x))

class FAI:
    def __init__(self, 
                 x:np.array, 
                 demand:np.array,
                 lr=1,
                 perishable=False,
                 h=1,b=3):
        '''
        

        Parameters
        ----------
        x : np.array
            x features shape:(T,N).
        demand : np.array
            demand shape:(T,1).
        perishable : TYPE, optional
            perishable or not. The default is False.
        h : TYPE, optional
            holding cost. The default is 1.
        b : TYPE, optional
            lost sales cost. The default is 3.
        theta : TYPE, optional
            a parameter influences learning rate. The default is 1.
        omega : TYPE, optional
            a list contains the upper and lower bound of the omega. The default is [0,1].

        Returns
        -------
        None.

        '''
        
        T,N = x.shape  
        # self.omega_low, self.omega_high = omega
        
        self.h = h
        self.b = b
        
        self.t = 0
        self.end = T
        self.perishable = perishable
        
        self.u = 0
        self.y = 0
        self.y_hat = 0
        
        self.x = x # shape: (T,N)
        self.demand = demand # shape: (T,1)
        self.z = np.random.uniform(low=0, high=1, size=(1,N)) # shape: (1,N)
        # self.z[0,:] = np.random.uniform(low=z1_low, high=z1_high, size=1) 
        # self.z = self.projection(self.z, self.omega_low, self.omega_high) # projection
        # self.epsilon = 1/((h+b)*theta)
        self.lr = lr
        
        self.gradient = 0
        
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        '''
        

        Raises
        ------
        StopIteration
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if self.t < self.end:
            self.y_hat = np.dot(self.z ,self.x[self.t].T)  # (1, 1)
            self.y = np.maximum(self.y_hat, self.u)
            # finish peirod 1, demand relized
            
            # update pram
            if self.y_hat-self.demand[self.t] > 0:
                gradient = self.h
            else:
                gradient = -self.b
            self.z -= (self.lr/(self.t+1))*gradient*self.x[self.t] # shape: (1,N)
            # self.z = self.projection(self.z, self.omega_low, self.omega_high) # projection
            self.t += 1
            return self.y.flatten()  
        else:
            raise StopIteration
    
    def projection(self, x, low, high):
        x[x<low] = low
        x[x>high] = high
        return x
        