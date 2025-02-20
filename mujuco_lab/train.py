import argparse
import yaml
import os
import wandb
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random
import configs.config_utils import load_config

# Model Load
from nn.A2C import A2C
from nn.REINFORCE import REINFORCE
from nn.policy import ContinuousPolicyNetwork


'''
train.py: Agent를 불러와서 agent.train() 수행하는 공간
- Agent를 불러와서, train()을 통해 학습
- wandb 적용 여부는 설정할 수 있게 조치
- config 내용을 바탕으로 모델명 짓기
- 학습 후의 결과는 results 폴더에 저장
'''


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    os.makedirs("results", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Config file path")
    parser.add_argument('--name', type=str, help='실험의 이름')
    args = parser.parse_args()
    config = load_config(args.config)
    
    wandb.init(project="mujuco_experiment", config=config, name=config["env_name"])
    
    env = gym.make(config["env_name"], render_mode="rgb_array")
    test_env = gym.make(config["env_name"], render_mode="rgb_array")
    
    if config['model'] == 'A2C':
        agent = A2C(env, test_env, config, DEVICE)
    elif config['model'] == 'REINFORCE':
        agent = REINFORCE(env, test_env, config, DEVICE)
    else:
        print(f"There is no model like {config['model']}")
    agent.train()
    
    # wandb.save(f"./results/{config['model']}_{config['method']}_{config['lr']}_{config['gamma']}.pth")
    env.close()

if __name__ == "__main__":
    main()
