import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).to(self.device))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).to(self.device))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).to(self.device))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).to(self.device))
        
        self.reset_parameters()  
        self.reset_noise()      

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features, device=self.device)
        epsilon_out = torch.randn(self.out_features, device=self.device)
        self.weight_epsilon = torch.outer(epsilon_out, epsilon_in)  
        self.bias_epsilon = epsilon_out

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    
class C51Network(nn.Module):
    def __init__(self, hidden_size, action_size, atoms=51, v_min=-10, v_max=10):
        super(C51Network, self).__init__()
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.support = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (atoms - 1)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 2, 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, hidden_size, 2, 1),
            nn.ReLU()
        )

        self.af = NoisyLinear(4 * hidden_size, action_size * atoms).to(self.device)
        self.vf = NoisyLinear(4 * hidden_size, atoms).to(self.device)  

    def forward(self, x, log=False):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x).view(batch_size, -1)

        a = self.af(x) 
        a = a.view(-1, self.action_size, self.atoms)
        v = self.vf(x)
        v = v.view(-1, 1, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        if log:
            return F.log_softmax(q, dim=2)
        else:
            return F.softmax(q, dim=2) 
    
    def get_action(self, x, available_actions=None):
        with torch.no_grad():
            probabilities = self.forward(x)
            q_values = (probabilities * self.support).sum(2) 
            
            for i in range(4):
                if available_actions[i] == 0:
                    q_values[0, i] = float('-inf')
            
            return torch.argmax(q_values, dim=1).item()

    def reset_noise(self):
        self.af.reset_noise()
        self.vf.reset_noise()

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv=False):
        super(Actor, self).__init__()
        self.conv = conv
        if conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, 3),
                nn.Tanh(),
                nn.Conv2d(64, hidden_dim, 2),
                nn.Tanh()
            )
        else:
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, hidden_dim)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        orthogonal_init(self.fc3)

    def forward(self, x):
        if self.conv:
            batch_size = x.shape[0]
            x = self.conv1(x).view(batch_size, -1)
        else:
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv=False):
        super(Critic, self).__init__()
        self.conv = conv
        if conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, 3),
                nn.Tanh(),
                nn.Conv2d(64, hidden_dim, 2),
                nn.Tanh()
            )
        else:
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, hidden_dim)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
        self.fc3 = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.fc3)

    def forward(self, x):
        if self.conv:
            batch_size = x.shape[0]
            x = self.conv1(x).view(batch_size, -1)
        else:
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
        return self.fc3(x)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  
        self.beta = beta 
        self.beta_increment = beta_increment  
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  
        weights = torch.FloatTensor(weights)

        experiences = [self.buffer[idx] for idx in indices]

        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta.flip(0):  
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    return torch.stack(advantage_list[::-1]).to(td_delta.device)

def normalize(x):
    mean = x.mean()
    std = x.std() + 1e-8
    return (x - mean) / std

class RunningMeanStd:
    def __init__(self, shape): 
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  
        self.gamma = gamma  
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  
        return x

    def reset(self):  
        self.R = np.zeros(self.shape)