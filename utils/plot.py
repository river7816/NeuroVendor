# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:29:53 2023

@author: river
"""
import numpy as np
import matplotlib.pyplot as plt

def log(x:np.array):
    '''
    

    Parameters
    ----------
    x : np.array
        x: regret shape (num_sample, T).

    Returns
    -------
    x : TYPE
        if x > 0, return log(x)
        if x = 0 return 0
        if x < 0 return -log(x).

    '''
    x[x>0] = np.log(x[x>0])
    x[x<0] = -np.log(-x[x<0])
    x[x==0] = 0
    return x
    
def plot_error_bands(y:np.array, label:str, num_std:float = 2, types = None, take_log = False):
    '''
    

    Parameters
    ----------
    y : np.array
        axis-y.
    label : str
        label of the plot.

    Returns
    -------
    None.

    '''
    x = np.arange(1, y.shape[1]+1)
    if take_log:
        y = log(y)
        x = np.log(x)

    mean = y.mean(axis=0)
    plt.plot(x, mean, label=label)
    if types == 'min_max':
        min_value = y.min(axis=0)
        max_value = y.max(axis=0)
        plt.fill_between(x, min_value, max_value, alpha=0.2)
    elif types == 'std':
        std = y.std(axis=0)
        plt.fill_between(x, mean-num_std*std, mean+num_std*std, alpha=0.2)

def plot_std(y:np.array, label:str, take_log = False):
    '''
    

    Parameters
    ----------
    y : np.array
        axis-y.
    label : str
        label of the plot.

    Returns
    -------
    None.

    '''
    x = np.arange(1, y.shape[1]+1)
    if take_log:
        y = log(y)
        x = np.log(x)

    std = y.std(axis=0)
    plt.plot(x, std, label=label)

