# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:28:25 2023

@author: river
"""
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from utils.loss import cost
from utils.data import DataGen
from models.model import ModelCompiler, FAI, Net
from utils.weight_adjuest import adjust_weights
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import torch.nn as nn
import torch.optim as optim



def init_weights_uniform(m, Range):
    low, high = Range[0], Range[1]
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, low, high)
        nn.init.uniform_(m.bias, low, high)



def train(model, optimizer, data_loader, loss_func=cost, lr_schedule=None):
    '''
    

    Parameters
    ----------
    model : custumed model
          class ModelComier.
    optimizer : torch.optim
        eg. Adam or SGD.
    data_loader : pytorch data_loader
        data_loader will return (x, D).
    loss_func : pytho function, optional
        loss function. The default is cost.
    lr_schedule : TYPE, optional
        learning rate schedule. The default is None.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    Y = []
    Regret = []
    Mean = []
    result = {}

    for x, demand, optimal_y in data_loader:
        '''
        b: batch_size = 1
        N: deatures
        
        x shape (b, N)
        demand shape (b,1)
        optimal_y shape (b,1)
        '''
        
        y, loss, cost = model(x, demand)
        cost_optimal = loss_func(optimal_y, demand)
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if lr_schedule is not None:
            lr_schedule.step()
        
        regret = cost - cost_optimal
        
        Y.append(y.squeeze().item())
        Regret.append(regret.squeeze().item())
        Mean.append(np.array(Regret).mean())
        
        result['y'] = Y
        result['regret'] = Regret
        result['regret_mean'] = Mean
    return result
       
def auto_simulate(N, data_type, lr, N_all=20, T=2000, scale=6, 
                  hidden_size=5, num_sample=200, 
                  auto_lr=False, cpu=-1):
    '''
    

    Parameters
    ----------
    N : int
        the number of feature available.
    data_type : str
        type of the data.
    N_all : int, optional
        total number of the features. The default is 20.
    T : int, optional
        time peirod. The default is 2000.
    num_sample : int, optional
        number of the sample. The default is 200.
    lr_schedule : pytorch learning rate schedule, optional
        learning rate schedule. The default is None.
    cpu : int, optional
        if cpu = -1, use all threads
        if cpu = -2, use all threads except one
        if cpu = 0, use one threads. The default is -1.
    scale: flost, optional
        the std of the delta

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    data = DataGen(N=N, T=T, num_sample=num_sample)
    dataset = data.load_data(data_type=data_type, scale=scale)
    
    result_linear = list()
    result_net = list()
    
    if cpu == 0:
        for i in tqdm(range(num_sample)):
            model_linear = ModelCompiler(nn.Linear(in_features=N_all, out_features=1))
            model_net = ModelCompiler(Net(input_size=N_all, hidden_size=hidden_size))
            
            model_linear.train()
            model_net.train()
        
            optimizer_linear = optim.SGD(model_linear.parameters(), lr=lr)
            optimizer_net = optim.SGD(model_net.parameters(), lr=lr)
            
            result_linear.append(train(model_linear, optimizer_linear,  DataLoader(dataset[i]))['regret_mean'])
            result_net.append(train(model_net, optimizer_net,  DataLoader(dataset[i]))['regret_mean'])
    else:
        def func_linear(i):
            linear = nn.Linear(in_features=N_all, out_features=1, bias=False)
            # Range = adjust_weights(N=N, model = linear, 
            #                        x_min = 1, x_max = 2, 
            #                        min_value=min_demand, 
            #                        max_value=max_demand)
            # linear.apply(lambda m: init_weights_uniform(m, Range))
            model_linear = ModelCompiler(linear)            
            model_linear.train()    
            optimizer_linear = optim.SGD(model_linear.parameters(), lr=lr)
            if auto_lr:
                lr_lambda = lambda epoch: 1/(epoch+1)
                lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_linear, lr_lambda=lr_lambda)
            else:
                lr_scheduler = None
            return train(model_linear, optimizer_linear, DataLoader(dataset[i]), lr_schedule=lr_scheduler)['regret_mean']
        
        def func_net(i):        
            # net = nn.Sequential(
            #     nn.Linear(in_features=N_all, out_features=10),
            #     nn.ReLU(),
            #     nn.Linear(in_features=10, out_features=1)       
            #     )
            net = Net(input_size=N_all, hidden_size=hidden_size)
            # Range = adjust_weights(N=N, model = net, 
            #                        x_min = 1, x_max = 2, 
            #                        min_value=min_demand, 
            #                        max_value=max_demand)
            # net.apply(lambda m: init_weights_uniform(m, Range))      
            model_net = ModelCompiler(net)
            model_net.train()
            optimizer_net = optim.SGD(model_net.parameters(), lr=lr)
            if auto_lr:
                lr_lambda = lambda epoch: 1/(epoch+1)
                lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_net, lr_lambda=lr_lambda)
            else:
                lr_scheduler = None
            return train(model_net, optimizer_net,  DataLoader(dataset[i]), lr_schedule=lr_scheduler)['regret_mean']
        
        
        result_linear = Parallel(n_jobs=cpu,prefer="processes")(delayed(func_linear)(index) for index in tqdm(range(num_sample)))
        result_net = Parallel(n_jobs=cpu,prefer="processes")(delayed(func_net)(index) for index in tqdm(range(num_sample)))
    result_linear = np.array(result_linear)
    result_net = np.array(result_net)
    return {'FAI': result_linear, 'NV': result_net}      



def auto_simulate_2(N, data_type, models, N_all=20, T=2000, num_sample=200, lr_schedule=None, cpu=-1):
    '''
    Parameters
    ----------
    N : int
        the number of feature available.
    data_type : str
        type of the data.
    N_all : int, optional
        total number of the features. The default is 20.
    T : int, optional
        time peirod. The default is 2000.
    num_sample : int, optional
        number of the sample. The default is 200.
    lr_schedule : pytorch learning rate schedule, optional
        learning rate schedule. The default is None.
    cpu : int, optional
        if cpu = -1, use all threads
        if cpu = -2, use all threads except one
        if cpu = 0, use one threads. The default is -1.
    *models : tuple
        tuple of models to train.

    Returns
    -------
    dict
        dictionary of regret means for each model.
    '''
    data = DataGen(N=N, T=T, num_sample=num_sample)
    dataset = data.load_data(data_type=data_type)
    
    results = {}
    for model in models:
        results[model.__class__.__name__] = []
    
    if cpu == 0:
        for i in tqdm(range(num_sample)):
            for model in models:
                model_compiler = ModelCompiler(model)
                model_compiler.train()
                optimizer = optim.SGD(model_compiler.parameters(), lr=1e-2)
                results[model.__class__.__name__].append(train(model_compiler, optimizer,  DataLoader(dataset[i]))['regret_mean'])
            for model in models:
                results[model.__class__.__name__] = np.array(results[model.__class__.__name__])
    else:
        def func(i):
            regret_means = {}
            for model in models:
                model_compiler = ModelCompiler(model)
                model_compiler.train()
                optimizer = optim.SGD(model_compiler.parameters(), lr=1e-2)
                regret_means[model.__class__.__name__] = train(model_compiler, optimizer, DataLoader(dataset[i]))['regret_mean']
            return regret_means
        
        result = Parallel(n_jobs=cpu)(delayed(func)(index) for index in tqdm(range(num_sample)))
        for model in models:
            model_name = model.__class__.__name__
            results[model_name] = np.array([r[model_name] for r in result])
    
    return results

        
def manual_simulate(N, data_type, lr, N_all=20, T=365, num_sample=200, cpu=-1, scale=6):
    '''
    

    Parameters
    ----------
    N : int
        feature available.
    data_type : str
        type of the data.
    N_all : int, optional
        total features. The default is 20.
    T : int, optional
        T period. The default is 720.
    num_sample : int, optional
        sample of the data. The default is 200.
    cpu : int, optional
        if cpu = -1, use all threads
        if cpu = -2, use all threads except one
        if cpu = 0, use one threads. The default is -1.

    Returns
    -------
    np.array
        regret shape: (num_sample, T).

    '''
    data = DataGen(N=N, T=T, N_all=N_all,num_sample=num_sample)
    data = data.prepare_data(data_type=data_type, scale=scale)

    def func(index):
        x, demand = data['x'][index], data['demand'][index], 
        y = np.array([y for y in FAI(x, demand, lr)])
        regret = cost(y, demand).numpy() # calculate the regret
        regret_mean = np.cumsum(regret)/(np.arange(regret.shape[0]) + 1) # calculate the mean regret before t
        return regret_mean

    result = Parallel(n_jobs=-1)(delayed(func)(index) for index in tqdm(range(200)))
    result = np.array(result)
    return {'FAI': result}
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
