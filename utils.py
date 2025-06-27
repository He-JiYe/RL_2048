import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
    def __init__(self, input_size, hidden_size, action_size, atoms=51, v_min=-10, v_max=10):
        super(C51Network, self).__init__()
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.support = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (atoms - 1)
        
        self.fc1 = NoisyLinear(input_size, 64).to(self.device)
        self.fc2 = NoisyLinear(64, 128).to(self.device)
        self.fc3 = NoisyLinear(128, hidden_size).to(self.device)

        self.af = NoisyLinear(hidden_size, action_size * atoms).to(self.device)
        self.vf = NoisyLinear(hidden_size, atoms).to(self.device)  

    def forward(self, x, log=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

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
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def _init_weights(self):
        init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        init.orthogonal_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
        return x
    
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def _init_weights(self):
        init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu')) 
        init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu')) 
        init.orthogonal_(self.fc3.weight, gain=nn.init.calculate_gain('relu')) 
        init.orthogonal_(self.fc4.weight, gain=nn.init.calculate_gain('Linear')) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DoubleCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DoubleCritic, self).__init__()
        self.critic1 = Critic(input_dim, hidden_dim)
        self.critic2 = Critic(input_dim, hidden_dim)

    def forward(self, state):
        value1 = self.critic1(state)
        value2 = self.critic2(state)
        return value1, value2
    
    def get_min_value(self, state):
        value1, value2 = self.forward(state)
        return torch.min(value1, value2)
    
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