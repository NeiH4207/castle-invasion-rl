#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim

from models.NoisyLayer import NoisyLinear


class BaseNet(nn.Module):

    def __init__(self):
        pass

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        pass
    
    def add_loss(self, loss):
        self.loss_history = np.append(self.loss_history, loss)
    
    def get_loss(self):
        return self.loss_history
    
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()		
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()		
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()		
        else:
            raise ValueError("Loss function not found")
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        x = x.reshape(-1, self.observation_shape)
        output = self.forward(x)
        return output.detach().cpu().numpy()
    
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)	
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)		
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)		
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
    
    def save(self, path=None):
        torch.save(self.state_dict(), path)
        
    def load(self, path=None, device=None):
        if path is None:
            raise ValueError("Path is not defined")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))
