# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:26:02 2023

@author: river
"""
import torch
import torch.nn.functional as F
import numpy as np

def cost(y, demand, h=1, b=3):
    '''
    

    Parameters
    ----------
    y : torch.tensor
        shape (1, 1).
    demand : torch.tensor
        shape (1, 1).
    h : TYPE, optional
        holding cost. The default is 1.
    b : TYPE, optional
        lost sales cost. The default is 3.

    Returns
    -------
    cost : TYPE
        cost of this period.

    '''
    if isinstance(y, np.ndarray) or isinstance(demand,np.ndarray):
        y = torch.tensor(y)
        demand = torch.tensor(demand)
    cost = h*F.relu(y-demand) + b*F.relu(demand-y) # shape (1, 1)
    return cost