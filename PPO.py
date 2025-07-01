import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import Actor, Critic
from utils import compute_advantage, normalize
from game import Game2048
from tqdm import tqdm
import json

class PPOAgent:
    def __init__(self, game_size=4, actor_learning_rate=0.01, critic_learning_rate=0.001,
                 discount_factor=0.95, weight_decay=1e-5, epochs=3, lmbda=0.95, eps=0.2):
        self.game_size = game_size
        self.state_size = game_size * game_size
        self.action_size = 4
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.weight_decay = weight_decay
        self.discount_factor = discount_factor
        self.epochs = epochs
        self.lmbda = lmbda
        self.eps = eps
        self.batch_size = 64
        self.model_path = 'model/ppo_agent_model.pth'
        
        self.conv = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.actor = Actor(self.state_size, 128, self.action_size, self.conv).to(self.device)
        self.critic = Critic(self.state_size, 128, self.conv).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate, weight_decay=self.weight_decay, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=self.weight_decay, eps=1e-5)
        scheduler1 = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, 0.5, 0.1, 10000)
        scheduler2 = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer, 0.5, 0.1, 10000)
        self.scheduler = [scheduler1, scheduler2]

        if os.path.exists(self.model_path):
            self.load_model()

    def preprocess_state(self, state):
        log_state = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])
        if np.max(log_state) > 0:
            log_state = log_state / 11.0  
        if self.conv:
            return np.expand_dims(log_state, 0)
        else:
            return log_state.flatten()

    def choose_action(self, state, available_actions):
        if not any(available_actions):
            return None
        
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_tensor) 
            probs = probs * torch.tensor(available_actions, dtype=torch.long, device=self.device)
            probs = probs / probs.sum()  
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).item()
    
    def sample_batches(self, transition_dict):
        total_size = len(transition_dict['states'])
        indices = torch.randperm(total_size).to(self.device)
        
        for start in range(0, total_size, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            
            yield batch_indices

    def replay(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(transition_dict['log_probs'], dtype=torch.float32).to(self.device)
        mask = torch.tensor(transition_dict['mask'], dtype=torch.long, device=self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze(1)  
            next_values = self.critic(next_states).squeeze(1)  

        td_target = rewards + self.discount_factor * next_values * (1 - dones)
        td_error = td_target - values
        advantage = compute_advantage(self.discount_factor, self.lmbda, td_error)
        advantage = normalize(advantage)  

        actor_losses, critic_losses, entropys = [], [], []
        for _ in range(self.epochs):
            for batch in self.sample_batches(transition_dict):
                states_b = states[batch]
                actions_b = actions[batch]
                advantages_b = advantage[batch]
                old_log_probs_b = old_log_probs[batch]
                td_targets_b = td_target[batch]
                mask_b = mask[batch]

                probs = self.actor(states_b)
                probs = probs * mask_b
                probs = probs / probs.sum(1, keepdim = True)
                dist = Categorical(probs)
                log_probs = dist.log_prob(actions_b)
                ratio = torch.exp(log_probs - old_log_probs_b)
                
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantages_b

                entropy = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - 0.005 * entropy
                
                values = self.critic(states_b).squeeze(1)  
                critic_loss = F.mse_loss(values, td_targets_b.detach())

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                for scheduler in self.scheduler:
                    scheduler.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropys.append(entropy.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropys)

    def train(self, episodes=1000, epochs=10):
        game = Game2048(self.game_size)
        result = {"return": [], "max_tile": [], "board_score": [],
                  "turns": [], "actor_loss": [], "critic_loss": [], 'entropy':[] }

        for i in range(epochs):
            with tqdm(total=int(episodes/10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(episodes/10)):
                    episode_return = 0
                    # 每幕游戏记录轨迹
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'log_probs': [], 'mask': []}
                    state = game.reset()
                    done = False
                    turn = 0
                    while not done:
                        turn += 1
                        available_actions = game.get_available_actions()
                        if not any(available_actions):
                            break
                        action, log_prob = self.choose_action(state, available_actions)
                        next_state, reward, done, info = game.move(action)

                        transition_dict['states'].append(self.preprocess_state(state))
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(self.preprocess_state(next_state))
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        transition_dict['log_probs'].append(log_prob)
                        transition_dict['mask'].append(available_actions)

                        state = next_state
                        episode_return += reward

                    for key, value in transition_dict.items():
                        transition_dict[key] = np.array(value)
                    actor_loss, critic_loss, entropy = self.replay(transition_dict)  

                    result["return"].append(episode_return)
                    result["max_tile"].append(game.get_max_tile().item())
                    result["board_score"].append(game.score)
                    result["turns"].append(turn)  
                    result['actor_loss'].append(actor_loss)
                    result['critic_loss'].append(critic_loss)
                    result['entropy'].append(entropy)

                    if (i_episode+1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (episodes/10 * i + i_episode+1), 'return': '%.1f' % np.mean(result["return"][-10:]), 'board score': '%.1f' % np.mean(
                            result['board_score'][-10:]), 'max tile': '%.1f' % np.mean(result['max_tile'][-10:]), 'entropy': '%.3f' % np.mean(result['entropy'][-10:]), 'lr': '%.5f' % self.scheduler[0].get_last_lr()[0]})
                    pbar.update(1)

            self.save_model()  
            with open("result.json", "w", encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        self.save_model()
        with open("result.json", "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        self.draw(result, save=True)
        return result

    def play_move(self, game, return_q_values=False):
        state = game.get_state()
        available_actions = game.get_available_actions()
        
        if not any(available_actions):
            return None if not return_q_values else (None, [0, 0, 0, 0])
        
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.actor(state_tensor)  
            mask = probs * torch.tensor(available_actions, dtype=torch.long, device=self.device)

        action = mask.argmax()

        if return_q_values:
            return action, probs.squeeze(0).cpu().numpy()
        return action

    def save_model(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, self.model_path)
    
    def load_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"模型已加载.")
        except Exception as e:
            print(f"加载模型失败: {e}")