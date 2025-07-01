import numpy as np
import os
import torch
import torch.optim as optim
from utils import PrioritizedReplayBuffer, C51Network
from collections import deque
from game import Game2048

class QLearningAgent:
    def __init__(self, game_size=4, learning_rate=0.001, discount_factor=0.95, n_step=3,
                 atoms=51, v_min=-10, v_max=10):
        self.game_size = game_size
        self.state_size = game_size * game_size
        self.action_size = 4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = 64
        self.n_step = n_step
        self.memory = PrioritizedReplayBuffer(capacity=500000)
        self.n_step_buffer = deque(maxlen=n_step)
        self.model_path = 'model/rainbow_dqn_agent_model.pth'
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.model = C51Network(128, self.action_size, 
                               atoms=atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.target_model = C51Network(128, self.action_size, 
                                      atoms=atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        if os.path.exists(self.model_path):
            self.load_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess_state(self, state):
        log_state = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])

        if np.max(log_state) > 0:
            log_state = log_state / 11.0  
        return log_state
    
    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            
            reward = r + self.discount_factor * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return reward, next_state, done

    def remember(self, state, action, reward, next_state, done):
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            state = self.n_step_buffer[0][0]  
            action = self.n_step_buffer[0][1]  

            self.memory.add(state, action, n_step_reward, n_step_next_state, n_step_done)
    
    def choose_action(self, state, available_actions):
        if np.sum(available_actions) == 0:
            return None

        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.model.get_action(state_tensor, available_actions)
    
    def project_distribution(self, rewards, dones, next_probs):
        batch_size = rewards.size(0)

        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        next_support = rewards + (1 - dones) * (self.discount_factor ** self.n_step) * self.support.view(1, -1)
        next_support = torch.clamp(next_support, self.v_min, self.v_max)

        b = (next_support - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long()\
            .unsqueeze(1).expand(batch_size, self.atoms).to(self.device)
        
        proj_dist = torch.zeros(next_probs.size(), device=self.device)
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1),
            (next_probs * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1),
            (next_probs * (b - l.float())).view(-1)
        )
        
        return proj_dist
    
    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states = states.unsqueeze(1).to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.unsqueeze(1).to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        with torch.no_grad():
            next_actions = self.model(next_states).mean(2).max(1)[1]
            next_probs = self.target_model(next_states)[range(self.batch_size), next_actions]

        target_probs = self.project_distribution(rewards, dones, next_probs)

        current_probs = self.model(states)
        current_probs = current_probs[range(self.batch_size), actions]

        loss = -(target_probs * torch.log(current_probs + 1e-6)).sum(1)
        weighted_loss = (weights * loss).mean()

        priorities = loss.detach().cpu().numpy() + 1e-6  
        self.memory.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        self.model.reset_noise()  
        self.target_model.reset_noise()

        return weighted_loss.item()
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.model_path)
    
    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.update_target_model()
            print(f"模型已加载.")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    def train(self, episodes=1000, max_steps=50000, save_interval=100, target_update_interval=10, callback=None):
        self.model.train()

        game = Game2048(self.game_size)
        scores = []
        max_tiles = []
        losses = []
        rewards = []
        
        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0
            episode_loss = []
            episode_rewards = []
            
            while game.get_max_tile() < 1024 and episode > 100 and episode % 3 == 0:
                self.model.eval()
                available_actions = game.get_available_actions()
                if np.sum(available_actions) == 0:
                    state = game.reset()

                action = self.choose_action(state, available_actions)
                state, reward, done, info = game.move(action)
                if done:
                    state = game.reset()

            for step in range(max_steps):
                self.model.train()

                available_actions = game.get_available_actions()
                if np.sum(available_actions) == 0:
                    break
                
                action = self.choose_action(state, available_actions)
                next_state, reward, done, info = game.move(action)
                
                self.remember(state, action, reward, next_state, done)

                if len(self.memory.buffer) >= self.batch_size:
                    loss = self.replay()
                    if loss is not None:
                        episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                episode_rewards.append(reward)
                
                if done:
                    break

            scores.append(game.score)
            max_tiles.append(game.get_max_tile())
            if episode_loss:
                losses.append(np.mean(episode_loss))
            if episode_rewards:
                rewards.append(np.mean(episode_rewards))

            if episode % target_update_interval == 0:
                self.update_target_model()

            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_max_tile = np.mean(max_tiles[-10:])
                avg_loss = np.mean(losses[-10:]) if losses else 0
                avg_reward = np.mean(rewards[-10:]) if rewards else 0
                print(f"Episode: {episode}, 平均分数: {avg_score:.2f}, 平均最大数字: {avg_max_tile:.2f}, 平均损失: {avg_loss:.4f}, 平均奖励: {avg_reward:.4f}")

                if callback:
                    callback(episode, avg_score, avg_max_tile, avg_loss, avg_reward)
            
            if episode % save_interval == 0:
                self.save_model()

        self.save_model()
        print("训练完成！")
        
        return scores, max_tiles
    
    def play_move(self, game, return_q_values=False):
        self.model.eval()

        state = game.get_state()
        available_actions = game.get_available_actions()
        
        if np.sum(available_actions) == 0:
            return None if not return_q_values else (None, [0, 0, 0, 0])

        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if return_q_values:
            with torch.no_grad():
                probabilities = self.model(state_tensor)
                q_values = (probabilities * self.support).sum(2).cpu().numpy()[0]

            masked_q_values = q_values.copy()
            for i in range(4):
                if available_actions[i] == 0:
                    masked_q_values[i] = float('-inf')
            
            action = np.argmax(masked_q_values)
            return action, q_values
        return self.model.get_action(state_tensor, available_actions)
    