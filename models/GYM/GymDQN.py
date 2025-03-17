#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from models.BaseNet import BaseNet

from models.NoisyLayer import NoisyLinear


class CartPole(BaseNet):

    def __init__(
        self, 
        observation_shape: int, 
        n_actions: int,
        atom_size: int, 
        v_min: int,
        v_max: int,
        optimizer: str = "adamw",
        lr: float = 0.001,
        device='cuda') -> None:
        super(BaseNet, self).__init__()
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)
        self.feature = nn.Linear(observation_shape, 1024)
        
        # set advance layer
        self.advance_hidden = NoisyLinear(1024, 256)
        self.advance = NoisyLinear(256, n_actions * atom_size)
        
        # set value layer
        self.value_hidden = NoisyLinear(1024, 256)
        self.value = NoisyLinear(256, atom_size)
        
        
        self.set_optimizer(optimizer, lr)
        self.loss_history = np.array([])


    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature(x)
        adv_hid = F.relu(self.advance_hidden(feature))
        val_hid = F.relu(self.value_hidden(feature))
        
        advance = self.advance(adv_hid).view(
            -1, self.n_actions, self.atom_size
        )
        value = self.value(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advance - advance.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advance.reset_noise()
        self.value.reset_noise()
