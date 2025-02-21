import os
import tqdm
import wandb
import numpy as np
import torch
from collections import deque
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .policy import ContinuousPolicyNetwork

'''
REINFORCE.py: 
- REINFORCE 모델 구현
- Q-network는 q-network.py에 구현되어 있으며, Policy는 policy.py에 구현
- experience(), train(), test() 등이 class 함수로 구현현   
'''

class REINFORCE:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device
        self.method = config['method']
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.gamma = config['gamma']  # 할인율 (discount factor)
        self.total_episodes = config['total_episodes']
        self.replay_buffer = deque(maxlen=config["buffer_size"])

        # policy
        self.policy = ContinuousPolicyNetwork(self.state_dim, self.action_dim).to(self.device) 
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])

        # best param
        self.best_score = 0.0
        self.best_param = dict()

        # model name
        self.base_dir = f"mujuco_lab/results/{config['model']}"
        self.model_name = f"{config['model']}_{config['method']}_{config['lr']}_{config['gamma']}"

    def get_action(self, state):
        # 상태로부터 행동 샘플링
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob

    def train(self):
        rewards_history = []

        for episode in tqdm.tqdm(range(self.total_episodes)):
            obs, _ = self.env.reset()
            state = np.array(obs, dtype=np.float32)
            episode_reward = 0.0
            done = False
            log_probs = []  # 로그 확률 기록
            rewards = []  # 보상 기록
            while not done:
                action, log_prob = self.get_action(state)  # 행동과 로그 확률
                next_obs, reward, done, truncated, _ = self.env.step(action)
                next_state = np.array(next_obs, dtype=np.float32)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                log_probs.append(log_prob)  # 행동에 대한 로그 확률 저장
                rewards.append(reward)  # 보상 저장
                if done or truncated:
                    break
            
            # 누적 보상 계산
            discounted_rewards = self.compute_discounted_rewards(rewards)

            # 정책 업데이트
            loss = self.update_policy(log_probs, discounted_rewards)
            
            # episode reward 저장 및 best일 경우, 해당 policy param 저장장
            rewards_history.append(episode_reward)
            if episode_reward > best_score:
                best_param = self.policy.state_dict()

            wandb.log({"episode": episode, "reward": episode_reward})
            wandb.log({"episode": episode, "loss": loss.item()})
            print(f"Episode: {episode}, Reward: {episode_reward}")
        
        # Best 모델과 마지막 모델 각각 저장
        self.save_model(self.best_param, self.model_name + "_best.pth")
        self.save_model(self.policy.state_dict(), self.model_name + ".pth")
        
        return rewards_history

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards

    def update_policy(self, log_probs, discounted_rewards):
        # 기대 보상 (G) 계산
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs)  # 로그 확률을 텐서로 변환
        
        # 손실 함수 계산
        loss = -torch.sum(log_probs * discounted_rewards)
        
        # 정책 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def save_model(self, param, file_name):
        # `project/results` 디렉토리가 없으면 생성
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        # 모델의 state_dict를 저장합니다.
        file_path = os.path.join(self.base_dir, file_name)
        torch.save(param, file_path)

    def load_model(self, file_name):
        file_path = os.path.join(self.base_dir, file_name)
        # 모델을 로드하고 state_dict를 업데이트합니다.
        self.policy.load_state_dict(torch.load(file_path))
        self.policy.eval()  # 평가 모드로 전환



