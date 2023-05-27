# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:57:50 2023

@author: river
"""
import numpy as np
from tqdm import tqdm
from statistics import NormalDist
from torch.utils.data import Dataset
from scipy.stats import truncnorm
import torch

initial_pram = {
    'h':1,
    'b':3,
    'x_low':1,
    'x_high':2,
    'omega_low':5,
    'omega_high':15,
    'loc': 0 # mean of the delta
    # 'scale': 6, # std of the delta
    }



#%%


class DataGen:
    def __init__(self, N:int, T:int, num_sample:int, N_all:int=20):
        self.N = N
        self.N_all = N_all
        self.T = T
        self.num_sample = num_sample
        
    def prepare_data(self, data_type:str='linear', scale=6, prams=initial_pram):
        '''
        

        Parameters
        ----------
        D_t = g(\omega \times x_t) + delta_t
        
        num_sample : int
            number of sample path.
        data_type : str, optional
            linear, polynomial, exponential. The default is 'linear'.
        power : int, optional
            power of the g. The default is 2.
        prams : TYPE, optional
            prameters. The default is initial_pram.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        '''
        
        print('-----Begin Simulation-----')
        b,h,x_low,x_high,omega_low,omega_high,loc = prams.values()
        delta_low, delta_high = -3*scale, 3*scale
        N, N_all, T, num_sample = self.N, self.N_all,self.T, self.num_sample
        
        # z_range, z1_range = self.compute_z_range(N, N_all, scale, x_low, x_high, omega_low, omega_high)
        
        x = np.random.uniform(low=x_low, high=x_high, size=(num_sample,T,N_all)) # shape (num_sample, T, N)
        x[:,:,0] = 1
        omega = np.random.uniform(low=omega_low, high=omega_high, size=(num_sample, 1, N_all)) # shape (num_sample, 1, N)
        
        

        
        a, b = (delta_low - loc) / scale, (delta_high - loc) / scale
        delta = truncnorm.rvs(a, b, loc, scale, size=(num_sample, T, 1))
        theta = truncnorm.pdf(delta, a, b, loc, scale).min()
        
        if data_type == 'linear':
            g_x = x
        elif data_type == 'polynomial':
            power = np.random.randint(low=1,high=5,size=(num_sample, N_all))
            power = np.expand_dims(power, 1).repeat(T, axis=1)
            g_x_polynomial = x**power
            g_x = g_x_polynomial
        elif data_type == 'exponential':
            power = np.random.randint(low=1,high=5,size=(num_sample, N_all))
            power = np.expand_dims(power, 1).repeat(T, axis=1)
            g_x_exp = power*np.exp(x)
            g_x = g_x_exp
        elif data_type == 'sin':
            power = np.random.randint(low=1,high=5,size=(num_sample, N_all))
            power = np.expand_dims(power, 1).repeat(T, axis=1)
            g_x_sin = power*np.sin(x)
            g_x = g_x_sin
        else:
            raise ValueError("data type must be linear, polynomial, trigonometic and exponential!")
        print('\n-----End Simulation-----\n')
        # g_x += scale
        g_x_omega = np.stack([np.dot(g_x[i], omega[i].T) for i in tqdm(range(num_sample), desc = data_type)], axis=0) # shape (num_sample, T, 1)
        D = g_x_omega + delta
        optimal_y = g_x_omega + truncnorm.ppf(b/(b+h), a, b, loc, scale)
        
        '''
        if N<N_all, we set x[:,:,N:] = 0
        for example:
            N = 1, N_all = 20,
            x = [1, 0, 0...0]
        '''
        if N < N_all:
            x[:,:,N:] = 0
            
        return {'demand':D, 'optimal_y':optimal_y, 'x':x, 'theta': theta}
    
    # def compute_z_range(self, N:int, N_all:int, 
    #                     scale:float, x_low:float, 
    #                     x_high:float, omega_low:float, 
    #                     omega_high:float, bios=5):
    #     '''
        

    #     Parameters
    #     ----------
    #     N : int
    #         number of features.
    #     N_all : int
    #         total features.
    #     scale : float
    #         std of the delta.
    #     x_low : float
    #         lower bound of the x.
    #     x_high : float
    #         higher bound of the x.
    #     omega_low : float
    #         lower bound of the omega.
    #     omega_high : float
    #         upper bound of the omega.
    #     bios : TYPE, optional
    #         the difference between the z and omega. The default is 5.

    #     Returns
    #     -------
    #     z_range : list
    #         z range.
    #     z1_range : list
    #         z1 range.

    #     '''
    #     z_low, z_high = omega_low-bios, omega_high+bios
    #     z1_low, z1_high = z_low-3*scale, z_high+3*scale
    #     z1_low_N, z1_high_N = z1_low-z_low*x_low*(N_all-N), z1_high+z_high*x_high*(N_all-N)
    #     z_range = [z_low, z1_high]
    #     z1_range = [z1_low_N, z1_high_N]
    #     return z_range, z1_range
    
    def load_data(self, data_type='linear', scale=6):
        '''
        

        Parameters
        ----------
        data_type : TYPE, optional
            type of the data. The default is 'linear'.

        scale: std of the delta
        Returns
        -------
        dataset : pytorch dataset
            (x, D).

        '''
        data_dict = self.prepare_data(data_type=data_type, scale=scale)
   
        dataset = []
        for i in range(self.num_sample):
            data = ModelDataset(demand=data_dict['demand'][i], 
                                      x=data_dict['x'][i], 
                                      optimal_y=data_dict['optimal_y'][i])
            dataset.append(data)
        return dataset
    
class ModelDataset(Dataset):
    def __init__(self, demand:np.array, x:np.array, optimal_y:np.array):
        '''
        

        Parameters
        ----------
        demand : np.array
            demand shape (T, 1).
        x : np.array
            x features shape (T, N).
        optimal_y : np.array
            optimal y shape (T, 1).

        Returns
        -------
        None.

        '''
        super().__init__()
        self.demand = demand 
        self.x = x
        self.optimal_y = optimal_y
    
    def __getitem__(self, index):
        demand = torch.tensor(self.demand[index,:], dtype=torch.float32)
        x = torch.tensor(self.x[index,:], dtype=torch.float32)
        optimal_y = torch.tensor(self.optimal_y[index,:], dtype=torch.float32)
        return x, demand, optimal_y
    
    def __len__(self):
        return self.demand.shape[0]
        

def cost(stretagy, demand, h=1, b=3, return_bios = False):
    '''
    Parameters
    ----------
    stretagy : numpy array
       your own inventary strategy.
    demand : numpy array
        optimal array.
    h : float, optional
        holding cost. The default is 1.
    b : float, optional
        lost sales cost. The default is 3.
    return_bios: Boolean, optional
        return bios or not. The default is False.

    Returns
    -------
    dict
        bios: stretagy-demand at t.
        cost of very sample data
    '''
    bios = np.stack([stretagy[i]-demand[i] for i in range(stretagy.shape[0])], axis=0)
    bios[bios>=0] *= h
    bios[bios<0] *= -b
    if return_bios:
        return {'bios':bios, 'cost':bios.mean(axis=1)}
    else:
        return bios.mean(axis=1)