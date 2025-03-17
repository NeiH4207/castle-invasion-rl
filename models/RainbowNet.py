#!/usr/bin/env python
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from models import config
import logging

from models.NoisyLayer import NoisyLinear
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, kernel_size=3, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
    
class RainbowNet(nn.Module):
    def __init__(self, 
                 observation_shape, 
                 n_actions, 
                 atom_size: int, 
                 v_min: int,
                 v_max: int,
                 optimizer='adamw', 
                 device='cuda',
                 lr=0.001,
                 version=1):
        super(RainbowNet, self).__init__()
        # game params
        self.config = config[str(version)]
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)
        self.in_channels = observation_shape[0]
        self.elo_history = np.array([1000])
        self.loss_history = np.array([])
        
        self.conv1 = nn.Conv2d(self.in_channels , self.config['conv1-num-filter'], kernel_size=self.config['conv1-kernel-size'], 
                               stride=self.config['conv1-stride'], padding=self.config['conv1-padding'])
        
        self.bn1 = nn.BatchNorm2d(self.config['conv1-num-filter'])
        
        for block in range(self.config['num-resblocks']):
            setattr(self, "res_%i" % block,ResBlock(self.config['conv1-num-filter'],
                                                    self.config['conv1-num-filter'],
                                                    self.config['resblock-kernel-size']))
        
        
        self.out_conv1_dim = int((self.observation_shape[1] - self.config['conv1-kernel-size'] \
            + 2 * self.config['conv1-padding']) / self.config['conv1-stride'] + 1)
        self.flatten_dim = self.config['conv1-num-filter'] * ((self.out_conv1_dim) ** 2)
        
        # set advance layer
        self.advance_hidden = NoisyLinear(self.flatten_dim, self.config['fc1-num-units'])
        self.advance = NoisyLinear(self.config['fc1-num-units'], self.n_actions * self.atom_size)
        
        self.value_hidden = NoisyLinear(self.flatten_dim, self.config['fc1-num-units'])
        self.value = NoisyLinear(self.config['fc1-num-units'], self.atom_size)
        
        self.set_optimizer(optimizer, lr)
        
    def get_device(self):
        # detect device for tensor operations
        device = self.conv1.weight.device
        return device
    
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature vector from input."""
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(self.config['num-resblocks']):
            x = getattr(self, "res_%i" % block)(x)
        return x.view(-1, self.flatten_dim)
            
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
    
    def get_elo(self):
        return self.elo_history[-1]
    
    def set_elo(self, ELO):
        self.elo_history = np.append(self.elo_history, ELO)
    
    def add_loss(self, loss):
        self.loss_history = np.append(self.loss_history, loss)
    
    def get_loss(self):
        return self.loss_history
    
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()		# Hàm loss là tổng bình phương sai lệch
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()		# Hàm loss là binary cross entropy, với đầu ra 2 lớp
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()		# Hàm BCE logit sau đầu ra dự báo có thêm sigmoid, giống BCE
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()		# Hàm L1 loss nhưng có đỉnh được làm trơn, khả vi với mọi điểm
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()		# Hàm tối ưu logistic loss 2 lớp của mục tiêu và đầu ra dự báo
        else:
            raise ValueError("Loss function not found")
    
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)		# Tối ưu theo gradient descent thuần túy
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)		# Phương pháp Adadelta có lr update
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)		# Phương pháp Adagrad chỉ cập nhật lr ko nhớ
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
    
    
    def predict(self, x):
        self.eval()
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32, device=self.get_device())
        x = x.reshape(-1, self.observation_shape[0], self.observation_shape[1], self.observation_shape[2])
        output = self.forward(x).detach()
        return output.detach().cpu().numpy()
    
    def predict_probs(self, x):
        output = self.predict(x)
        # minmax scaling
        output = (output - np.min(output)) / (np.max(output) - np.min(output))
        output = output ** 4
        # softmax
        output = np.exp(output) / np.sum(np.exp(output))
        return output
    
    def get_pi_values(self, x):
        output_value = self.predict(x)[0]

        # minmax scaling
        output = (output_value - np.min(output_value)) / (np.max(output_value) - np.min(output_value))
        output = output ** 4
        # softmax
        pi = np.exp(output) / np.sum(np.exp(output))
        value = np.argmax(pi * output_value)
        return pi, value
        
    
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)		# Tối ưu theo gradient descent thuần túy
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)		# Phương pháp Adadelta có lr update
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)		# Phương pháp Adagrad chỉ cập nhật lr ko nhớ
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
    
    def save(self, path=None, metadata=True):
        if path is None:
            logging.error('Model path not specified')
        state_dict = self.state_dict()
        if metadata:
            checkpoint = {
                'elo_history': self.elo_history,
                'loss_history': self.loss_history,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
            }
        else:
            checkpoint = {
                'state_dict': state_dict,
            }
        torch.save(checkpoint, path)
        logging.info('Model saved to {}'.format(path))
        
    def load(self, path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path is None:
            raise ValueError("Path is not defined")
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint['state_dict'])
        
        if 'elo_history' in checkpoint:
            self.elo_history = checkpoint['elo_history']
            
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        print('Model loaded from {}'.format(path))
