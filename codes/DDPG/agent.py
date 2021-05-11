#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2021-03-17 20:43:25
@Discription: 
@Environment: python 3.7.7
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.model import Actor, Critic
from common.memory import ReplayBuffer
import wandb

class DDPG:
    def __init__(self, n_states, n_actions, cfg):
        self.device = cfg.device
        self.critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
        # 最开始，让4个网络中对应的Actor和Critic网络的参数保持相同
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # torch.detach()用于切断反向传播
        action_value = action.detach().cpu().numpy()[0, 0]
        return action_value

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(
            self.batch_size)
        # 将所有变量转为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # 注意critic将(s_t,a)作为输入, 将当前的状态和actor决定的动作作为参数，传给critic，critic决定判断好坏
        actor_action = self.actor(state)    # shape (batch_size,)
        policy_loss = self.critic(state, actor_action)   #shape (batch_size)
        #
        policy_loss = -policy_loss.mean()
        # next_state , shape, [batch_size, state_num] torch.Size([128, 3]), 目标网络，用于Q网络参数
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
    def save(self,path):
        torch.save(self.target_net.state_dict(), path+'DDPG_checkpoint.pth')

    def load(self,path):
        self.actor.load_state_dict(torch.load(path+'DDPG_checkpoint.pth')) 