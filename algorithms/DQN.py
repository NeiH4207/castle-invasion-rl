from collections import deque
from copy import deepcopy
import numpy as np
from torch import optim as optim
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        
    def __len__(self):
        return len(self.memory)
        
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []
        
        for state, action, reward, next_state, done in batch:
            state_batch.append(state)
            action_batch.append([action])
            reward_batch.append([reward])
            next_state_batch.append(next_state)
            done_batch.append([done])
        
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)
           
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

class DQN():
    def __init__(self, observation_shape=None, n_actions=None, model=None,
                    tau=0.005, gamma=0.99, epsilon=0.9, epsilon_min=0.05, 
                    epsilon_decay=0.99,
                    memory_size=4096,  model_path=None):
        super().__init__()
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.memory = Memory(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_net = model
        self.target_net = deepcopy(model)
        self.model_path = model_path
        self.history = {
            'loss': [],
            'reward': []
        }
        self.counter = 0
        
    def fully_mem(self, perc=1.0):
        return len(self.memory) / (self.memory_size - 1) >= perc
    
    def reset_memory(self):
        self.memory.clear()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
        
    def get_action(self, state, valid_actions=None, model=None):
        if model is None:
            model = self.policy_net
        state = torch.FloatTensor(np.array(state)).to(self.device)
        act_values = model.predict(state)[0]
        
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                act_values[~valid_actions] = -float('inf')
                return np.random.choice(np.arange(self.n_actions)[valid_actions])
            else:
                return np.random.choice(np.arange(self.n_actions))
        return int(np.argmax(act_values))  # returns action
        
    def replay(self, batch_size, verbose=False):

        if len(self.memory) < batch_size:
            return 0 # memory is still not full
        if verbose:
            _tqdm = tqdm(range(len(self.memory) // batch_size + 1), desc='Replay')
        else:
            _tqdm = range(len(self.memory) // batch_size + 1)
        total_loss = 0
        self.policy_net.train()
        
        for i in _tqdm:
            minibatch = self.memory.sample(batch_size)
            state_batch = torch.Tensor(minibatch[0]).to(self.device) 
            action_batch = torch.LongTensor(minibatch[1]).to(self.device)
            reward_batch = torch.Tensor(minibatch[2]).to(self.device)
            next_state_batch =  torch.Tensor(minibatch[3]).to(self.device)
            done_batch = torch.Tensor(minibatch[4]).to(self.device)
                
            
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1)[0]
            
            expected_state_action_values = (1 - done_batch) * (next_state_values.unsqueeze(1) * self.gamma) + reward_batch

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.reshape(-1, 1))
            
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)
            # Optimize the model
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_net.optimizer.step()
        
            self.soft_update()
            total_loss += loss.item()
            mean_loss = total_loss / (i + 1)
            
        self.policy_net.add_loss(mean_loss)
        self.adaptiveEGreedy()
        if self.counter % int(1 / self.tau) == 0:
            self.hard_update()
            
        return self.policy_net.get_loss()
    
    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        self.counter += 1
        
    def hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def load_model(self, path, device=None):
        self.policy_net.load(path, device=device)
        self.target_net = deepcopy(self.policy_net)
        
    def save_model(self):
        self.target_net.save(self.model_path)
        
    def get_model(self):
        return self.policy_net
        
    def reset_history(self):
        self.history['loss'] = []