import os
import tqdm
import wandb
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .policy import ContinuousPolicyNetwork
from .network import Critic

'''
A2C.py: 
- Actor-Critic 모델 구현
- Q-network는 q-network.py에 구현되어 있으며, Policy는 policy.py에 구현
- experience(), train(), test() 등이 class 함수로 구현
- V, Q, TD, TD(lambda), Natural 등 다양한 방식의 파라미터 업데이트 구현   
'''

class A2C:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.method = config['method']

        # actor & critic
        self.policy = ContinuousPolicyNetwork(self.state_dim, self.action_dim).to(self.device) 
        self.vnet = Critic()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.critic_optimizer = optim.Adam(self.vnet.parameters(), lr=config['lr'])
        self.gamma = config['gamma']  # 할인율 (discount factor)
        self.lambda_ = config['lambda']
        self.total_episodes = config['total_episodes']
        self.replay_buffer = deque(maxlen=config["buffer_size"])

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

    def compute_fisher_information(self, states, actions, log_probs):
        """
        Fisher Information Matrix 계산 (FIM)
        """
        fisher_information = torch.zeros_like(self.policy.parameters())
        for state, action, log_prob in zip(states, actions, log_probs):
            grad_log_prob = torch.autograd.grad(log_prob, self.policy.parameters(), retain_graph=True)
            fisher_information += torch.tensor([torch.outer(grad, grad) for grad in grad_log_prob])
        return fisher_information

    def train(self):
        rewards_history = []

        for episode in tqdm.tqdm(range(self.total_episodes)):
            obs, _ = self.env.reset()
            state = np.array(obs, dtype=np.float32)
            episode_reward = 0.0
            done = False
            log_probs = []  # 로그 확률 기록
            rewards = []  # 보상 기록
            states = []  # 상태태 기록
            actions = []  # 행동 기록
            next_states = []  # 다음 상태 기록
            dones = []  # 종료여부 기록
            while not done:
                action, log_prob = self.get_action(state)  # 행동과 로그 확률
                next_obs, reward, done, truncated, _ = self.env.step(action)
                next_state = np.array(next_obs, dtype=np.float32)
                self.replay_buffer.append((state, action, reward, next_state, done))
                episode_reward += reward
                log_probs.append(log_prob)  # 행동에 대한 로그 확률 저장
                rewards.append(reward)  # 보상 저장
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                if done or truncated:
                    break
            
            # ----------------------------------------------------
            # 1) Critic target (여기서는 state_values라고 명명)
            #    method별로 다른 형태(Q, Advantage, TD, 등)를 계산
            # ----------------------------------------------------
            targets, eligibility_traces, fisher_inv = self.compute_targets(states, actions, rewards, next_states, dones, log_probs)

            # ----------------------------------------------------
            # 2) Policy update (Actor)
            #    -log πθ(s,a) * [ Q, Advantage, TD-error, ... ]
            # ----------------------------------------------------
            loss = self.update_policy(log_probs, targets, eligibility_traces)
            
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

    def compute_targets(self, states, actions, rewards, next_states, dones, log_probs):
        state_values = []
        eligibility_traces = []
        fisher_inv = []
        e_t = torch.zeros_like(states[0], dtype=torch.float32).to(self.device)  # 초기화: 기울기

        for state, action, next_state, reward, done, log_prob in zip(states, actions, next_states, rewards, dones, log_probs):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            log_prob = torch.tensor(log_prob, dtype=torch.float32).to(self.device)

            if self.method == 'Q':
                # Q-learning: Compute Q(s, a) for the current state-action pair
                next_v_value = self.vnet(next_state)
                q_value = reward + next_v_value
                state_values.append(q_value)

            elif self.method == 'Advantage':
                # Advantage: Compute Advantage (A = Q - V)
                v_s = self.vnet(state)
                v_s_next = self.vnet(next_state)
                advantage = reward + (1 - done) * self.gamma * v_s_next - v_s
                state_values.append(advantage)

            elif self.method == 'TD_lambda':
                # TD(lambda): Handle eligibility traces and updates here
                v_s = self.vnet(state)
                v_s_next = self.vnet(next_state)
                td_error = reward + (1 - done) * self.gamma * v_s_next - v_s
                state_values.append(td_error)
                # eligibility traces updated
                e = self.lambda_ * e +  log_prob
                eligibility_traces.append(e)

            elif self.method == 'Natural':
                # Natural Policy Gradient: Adjust the gradient step size using Fisher Information Matrix
                v_s = self.vnet(state)
                v_s_next = self.vnet(next_state)
                td_error = reward + (1 - done) * self.gamma * v_s_next - v_s
                state_values.append(td_error)

        if self.method == 'Natural':    
            # Fisher Information Matrix 계산
            fisher_information = self.compute_fisher_information(states, actions, log_probs)
            fisher_inv = torch.inverse(fisher_information + 1e-8 * torch.eye(fisher_information.size(0)))

        return state_values, eligibility_traces, fisher_inv

    def update_policy(self, log_probs, pg_targets, eligibility_traces, fisher_inv=False):
        """
        policy gradient update
         - 일반적인 형태:  Loss = - Σ [ logπ(s,a) * (Q / Advantage / TD-lambda / ...) ]
        """
        pg_targets_t = torch.tensor(pg_targets, dtype=torch.float32).to(self.device)
        eligibility_traces = torch.tensor(eligibility_traces, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs) 

        # 손실 함수
        # L = - Σ [ logπ(s,a) * pg_targets ]
        if self.method == "TD_lambda":
            loss = -torch.sum(eligibility_traces * pg_targets_t)
        elif self.method == 'Natural':
            # 자연적 그라디언트 계산
            loss = -torch.matmul(fisher_inv, log_probs * state_values)
        else:
            loss = -torch.sum(log_probs * pg_targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Critic도 업데이트 (Qnet)
        self.update_critic(pg_targets_t)

        return loss

    def update_critic(self):
        """
        Critic(V(s)) 업데이트
         - TD(0)라면 target = r + gamma * V(s') 로 MSE
         - 여기서는 replay_buffer에서 mini-batch를 뽑아 업데이트 예시
        """
        if len(self.replay_buffer) < 32:
            return

        batch_size = 32
        transitions = [self.replay_buffer[i] for i in range(batch_size)]
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        v_s = self.vnet(states_t).squeeze(-1)        # (batch,)
        v_s_next = self.vnet(next_states_t).squeeze(-1)  # (batch,)

        target_v = rewards_t + (1 - dones_t) * self.gamma * v_s_next
        critic_loss = F.mse_loss(v_s, target_v.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
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



