# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:50:55 2023

@author: river
"""
import torch
import torch.nn as nn
# Define a function to adjust the initial weights of a PyTorch model and output the range of weights
def adjust_weights(N, model, x_min, x_max, min_value, max_value, lr = 0.01, bias=500):
    weight_range = []
    # Input shape 
    input_shape = next(model.parameters()).size()[1]
    
    min_x = torch.full((input_shape, ), x_max, dtype=torch.float32)
    min_x[0] = torch.full((1, ), 1.)
    
    max_x = torch.full((input_shape, ), x_min, dtype=torch.float32)
    max_x[0] = torch.full((1, ), 1.)
    
    if N < input_shape:
        min_x[N:] = 0
        max_x[N:] = 0

    # Iterate through the model's parameters
    for param in model.parameters():
        param.data.fill_(0)
    pred_value = model(min_x)
    while pred_value.item() < min_value-bias:  
        for param in model.parameters():
            # Calculate the predicted value using the current weights
            with torch.no_grad():
                pred_value = model(min_x)

                # Decrease the weights to fit within the specified range
            param.data += lr
            
    weight_range.append(next(model.parameters()).data.mean().item())
    
    
    for param in model.parameters():
        param.data.fill_(0)
    pred_value = model(max_x)
    while pred_value.item() < max_value+bias:  
        for param in model.parameters():
            # Calculate the predicted value using the current weights
            with torch.no_grad():
                pred_value = model(max_x)

                # Decrease the weights to fit within the specified range
            param.data += lr
    
    weight_range.append(next(model.parameters()).data.mean().item())
    
    return weight_range
    
    
    
    
    
#%%   
# Call the adjust_weights function with a sample model, input, and weight range
net_model = nn.Sequential(
     nn.Linear(in_features=20, out_features=10),
     nn.ReLU(),
     nn.Linear(in_features=10, out_features=1)       
     )

linear_model = nn.Linear(in_features=20, out_features=1)

# Get the adjusted weight ranges
Range = adjust_weights(N=1, model= linear_model, x_min=1, x_max=2, min_value=150, max_value=200)









