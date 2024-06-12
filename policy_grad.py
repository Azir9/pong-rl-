import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.version
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(64,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(64,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            torch.nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(32,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            torch.nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(16,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            torch.nn.Conv2d(in_channels=16,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(8,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            torch.nn.Conv2d(in_channels=8,out_channels=4,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(4,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            torch.nn.Conv2d(in_channels=4,out_channels=2,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            torch.nn.BatchNorm2d(2,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            torch.nn.Flatten(),
            
            torch.nn.Linear(in_features=12,out_features=action_dim),
            #torch.nn.ReLU(),
            #torch.nn.Linear(in_features=32,out_features=action_dim),
            #torch.nn.ReLU()  
        )

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.net(x)
        return F.softmax(x, dim=1)
    
class REINFORCE:#
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
