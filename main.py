# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:03:17 2023

@author: river
"""
import matplotlib.pyplot as plt
from utils.online_train import auto_simulate, manual_simulate
from models.model import Net
import torch.nn as nn
from utils.plot import plot_error_bands,plot_std
import scienceplots

T = 365 # time peirod
typ = ['linear', 'polynomial', 'exponential', 'sin'] # type of the data
title = ['$g(x) = x$', '$g(x) = polynomial$','$g(x) = e^x$','$g(x) = sinx$'] # title of the picture

'''
auto_simulate use torch auto_grad

manual_simulate use numpy grad manually
'''
#%% main results
for typ, title in zip(typ, title):
    results = {n: auto_simulate(N = n, T=T, lr=1, data_type = typ, num_sample=200, auto_lr=True) for n in [1, 10, 20]}
    results_manual = {n: manual_simulate(N = n, T=T, lr=1, data_type = typ, num_sample=200) for n in [1, 10, 20]}
    
    plt.style.use(['science', 'no-latex', 'grid'])
    plt.figure(figsize=(4.5, 2.41), dpi=200)
    for model in results_manual[1].keys():
        for n in [1, 10, 20]:
            plot_error_bands(results_manual[n][model], label=f'{model} {n}', take_log = False)
    for model in ['NV']:
        for n in [1, 10, 20]:
            plot_error_bands(results[n][model], label=f'{model} {n}', take_log = False)
    plt.title(title)
    # plt.xlabel('time $\log {t}$')
    # plt.ylabel('Mean $\log{Regret_{t}}$')
    plt.xlabel('time $t$')
    plt.ylabel('Mean $Regret_{t}$')
    plt.legend()
    plt.savefig(f'picture/{typ}_mean_lr.png')
    
    plt.figure(figsize=(4.5, 2.41), dpi=200)
    for model in results_manual[1].keys():
        for n in [1, 10, 20]:
            plot_error_bands(results_manual[n][model], label=f'{model} {n}', take_log = True)
    for model in ['NV']:
        for n in [1, 10, 20]:
            plot_error_bands(results[n][model], label=f'{model} {n}', take_log = True)
    plt.title(title)
    plt.xlabel('time $\log {t}$')
    plt.ylabel('Mean $\log{Regret_{t}}$')
    # plt.xlabel('time $t$')
    # plt.ylabel('Mean $Regret_{t}$')
    plt.legend()
    plt.savefig(f'picture/{typ}_mean_lr_log.png')
    
    # plt.figure(figsize=(4.5, 2.41), dpi=200)
    # for model in ['FAI','Net']:
    #     for n in [1, 10, 20]:
    #         plot_std(results[n][model], label=f'{model} {n}', take_log = False)
    # # for model in results_manual[1].keys():
    # #     for n in [1, 10, 20]:
    # #         plot_std(results_manual[n][model], label=f'{model} {n}', take_log = False)
    # plt.title('Std ' + title)
    # # plt.xlabel('time $\log {t}}$')
    # # plt.ylabel('Std $\log{Regret_{t}}$')
    # plt.xlabel('time $t$')
    # plt.ylabel('Std $Regret_{t}$')
    # plt.legend()
    # plt.savefig(f'picture/{typ}_std_lr.png')
#%% different std, we neeed to set num_n - 1,10,20

# num_n = 1
# for typ, title in zip(typ, title):
#     results = {std: auto_simulate(N = num_n, T=T, lr=1, scale=std, data_type = typ, num_sample=200, auto_lr=True) for std in [6, 12, 24]}
#     results_manual = {std: manual_simulate(N = num_n, T=T, lr=1, scale=std, data_type = typ, num_sample=200) for std in [6, 12, 24]}
#     plt.style.use(['science', 'no-latex', 'grid'])
#     plt.figure(figsize=(4.5, 2.41), dpi=200)
#     for model in results_manual[6].keys():
#         for std in [6,12,24]:
#             plot_error_bands(results_manual[std][model], label=f'{model} Std {std}', take_log = True)
#     for model in ['Net']:
#         for std in [6,12,24]:
#             plot_error_bands(results[std][model], label=f'{model} Std {std}', take_log = True)
#     plt.title(f'{title} $n={num_n}$')      
#     plt.xlabel('time $\log {t}}$')
#     plt.ylabel('Std $\log{Regret_{t}}$')
#     plt.legend()
#     plt.savefig(f'picture_std/{typ}_mean_lr_{num_n}_log.png')
    
#%% different hidden unit size
hidden_unit = [2, 5, 7, 10, 12, 15]
for typ, title in zip(typ, title):
    results = {hidden_size: auto_simulate(N = 20, T=T, lr=1, data_type = typ, hidden_size=hidden_size, num_sample=200, auto_lr=True) for hidden_size in hidden_unit}
    results_manual = {20: manual_simulate(N = 20, T=T, lr=1, data_type = typ, num_sample=200)}
    plt.style.use(['science', 'no-latex', 'grid'])
    plt.figure(figsize=(4.5, 2.41), dpi=200)
    for hidden_size in hidden_unit:
        plot_error_bands(results[hidden_size]['NV'], label=f'NV S={hidden_size}', take_log = True)
    plot_error_bands(results_manual[20]['FAI'], label=f'FAI', take_log = True)
    plt.title(f'{title}')      
    # plt.xlabel('time $t$')
    # plt.ylabel('Mean $Regret_{t}$')
    plt.xlabel('time $\log {t}}$')
    plt.ylabel('Mean $\log{Regret_{t}}$')
    plt.legend()
    plt.savefig(f'picture_size/{typ}_differ_net_size_log.png')
    
    plt.figure(figsize=(4.5, 2.41), dpi=200)
    for hidden_size in hidden_unit:
        plot_error_bands(results[hidden_size]['NV'], label=f'NV S={hidden_size}', take_log = False)
    plot_error_bands(results_manual[20]['FAI'], label=f'FAI', take_log = False)
    plt.title(f'{title}')      
    plt.xlabel('time $t$')
    plt.ylabel('Mean $Regret_{t}$')
    # plt.xlabel('time $\log {t}}$')
    # plt.ylabel('Std $\log{Regret_{t}}$')
    plt.legend()
    plt.savefig(f'picture_size/{typ}_differ_net_size.png')


# for typ,title in zip(typ,title):
#     result_1 = manual_simluate(N=1, T=T, data_type=typ)
#     result_10 = manual_simluate(N=10, T=T, data_type=typ)
#     result_20 = manual_simluate(N=20, T=T, data_type=typ)   
#     plt.style.use(['science','no-latex','grid'])
#     plt.figure(figsize=(7.5, 5.625), dpi=200)
#     plot_error_bands(result_1, label='linear 1')
#     plot_error_bands(result_10, label='linear 10')
#     plot_error_bands(result_20,label='linear 20')
#     plt.title(title)
#     plt.xlabel('log(time) $t$')
#     plt.ylabel('log(regret)')
#     plt.legend()
#     plt.savefig('picture/{}_1.png'.format(typ)) 
