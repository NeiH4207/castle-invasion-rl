from typing import Dict
import numpy as np
from torch import optim as optim
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.DQN import DQN
from src.priority import PrioritizedReplayBuffer, ReplayBuffer

class Rainbow(DQN):
    
    def __init__(
        self, 
        observation_shape=None,
        n_actions=None, 
        model=None,
        # DQN parameters
        tau=0.005, 
        gamma=0.99, 
        epsilon=0.25, 
        epsilon_min=0.01, 
        epsilon_decay=0.995, 
        memory_size=4096, 
        batch_size=32, 
        model_path=None,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # N-step Learning
        n_step: int = 3,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 500.0,
        atom_size: int = 51,
        ):

        super().__init__(observation_shape, n_actions, model, tau, 
                         gamma, epsilon, epsilon_min, epsilon_decay,
                         memory_size, model_path)
        self.memory = PrioritizedReplayBuffer(
            observation_shape, memory_size, batch_size, alpha
        )
        self.beta = beta
        self.prior_eps = prior_eps
        self.n_step = n_step
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)
        self.transition = list()
        self.memory_n = ReplayBuffer(
                observation_shape, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
        
    def set_multi_agent_env(self, n_agents):
        self.n_agents = n_agents
        
        self.memory = PrioritizedReplayBuffer(
            self.observation_shape, 
            self.memory_size, 
            self.batch_size, 
            self.alpha,
            n_step=self.n_step * n_agents + 1,
            n_agents=n_agents
        )
        
        self.memory_n = ReplayBuffer(
            self.observation_shape, 
            self.memory_size, 
            self.batch_size, 
            n_step=self.n_step * n_agents + 1, 
            gamma=self.gamma,
            n_agents=n_agents
        )
    def reset_memory(self):        
        self.memory.size = 0
        
    def get_action(self, state, valid_actions=None, model=None):
        epsilon = self.epsilon
        if model is None:
            model = self.policy_net
            epsilon = 0
        if np.random.rand() <= epsilon:
            if valid_actions is None or sum(valid_actions) == 0:
                return np.random.choice(self.n_actions)
            return np.random.choice(np.arange(self.n_actions)[valid_actions])
        else:
            state = torch.FloatTensor(np.array(state)).to(self.device)
            act_values = model.predict(state)[0]
            if valid_actions is not None:
                act_values[~valid_actions] = -float('inf')
            return int(np.argmax(act_values))  # returns action
        
    def calculate_dqn_loss(
        self, 
        samples: Dict[str, np.ndarray], 
        gamma: float
    ) -> torch.Tensor:
            state_batch = Variable(torch.FloatTensor(samples['obs'])).to(self.device)
            action_batch = Variable(torch.LongTensor(samples['acts'])).to(self.device)
            next_state_batch = Variable(torch.FloatTensor(samples['next_obs'])).to(self.device)
            reward_batch = Variable(torch.FloatTensor(samples['rews'])).to(self.device)
            done_batch = Variable(torch.FloatTensor(samples['done'])).to(self.device)
            
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
            batch_size = state_batch.size(0)
            
            with torch.no_grad():
                next_action_batch = self.policy_net(next_state_batch).argmax(1)
                next_dist = self.target_net.dist(next_state_batch)
                next_dist = next_dist[range(batch_size), next_action_batch]
            
                t_z = reward_batch.reshape(-1, 1) + (1 - done_batch).reshape(-1, 1) * gamma \
                    * self.policy_net.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (
                    torch.linspace(
                        0, (batch_size - 1) * self.atom_size, batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(batch_size, self.atom_size)
                    .to(self.device)
                )

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

            dist = self.policy_net.dist(state_batch)
            log_p = torch.log(dist[range(batch_size), action_batch])
            elementwise_loss = -(proj_dist * log_p).sum(1)
            return elementwise_loss
    
    def replay(self, batch_size, verbose=False):

        if len(self.memory) < batch_size:
            return 0 
        
        if verbose:
            _tqdm = tqdm(range(len(self.memory) // batch_size + 1), desc='Replay')
        else:
            _tqdm = range(len(self.memory) // batch_size + 1)
        total_loss = 0
        self.policy_net.train()
        
        for i in _tqdm:
            samples = self.memory.sample_batch(self.beta)
            weight_batch = Variable(torch.FloatTensor(samples['weights'])).to(self.device)
            elementwise_loss = self.calculate_dqn_loss(samples, self.gamma)
            loss = torch.mean(elementwise_loss * weight_batch)
            
            n_step_samples = self.memory_n.sample_batch_from_idxs(samples['indices'])
            gamma = self.gamma ** self.n_step
            n_elementwise_loss_loss = self.calculate_dqn_loss(n_step_samples, gamma)
            n_loss = torch.mean(n_elementwise_loss_loss * weight_batch)
            loss += n_loss
                # Optimize the model
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.policy_net.optimizer.step()
            # PER: update priorities
            self.soft_update()
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(samples['indices'], new_priorities)
            
            total_loss += loss.item()
            mean_loss = total_loss / (i + 1)
            
        self.policy_net.add_loss(mean_loss)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        self.adaptiveEGreedy()
        return self.policy_net.get_loss()
        
    
    def update_beta(self):
        self.beta = min(1.0, self.beta + 1e-4)
        