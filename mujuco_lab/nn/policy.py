import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ContinuousPolicyNetwork, self).__init__()
        self.max_action = max_action
        
        # 네트워크의 층 정의
        self.fc1 = nn.Linear(state_dim, 256)  # 첫 번째 FC layer
        self.fc2 = nn.Linear(256, 256)        # 두 번째 FC layer
        self.fc3_mean = nn.Linear(256, action_dim)  # 행동 평균을 위한 FC layer
        self.fc3_log_std = nn.Linear(256, action_dim)  # 로그 표준편차를 위한 FC layer

    def forward(self, state):
        # 상태를 네트워크에 통과시키며 각 층을 거침
        x = F.relu(self.fc1(state))  # 첫 번째 층
        x = F.relu(self.fc2(x))      # 두 번째 층
        
        # 평균과 로그 표준편차 출력
        mean = self.fc3_mean(x)  # 평균 값
        log_std = self.fc3_log_std(x)  # 로그 표준편차 값
        
        # 로그 표준편차를 표준편차로 변환
        std = torch.exp(log_std)  
        
        # 이 값을 사용해 정규 분포의 파라미터를 반환
        return mean, std

