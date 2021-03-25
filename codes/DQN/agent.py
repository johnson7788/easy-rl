#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2021-03-13 14:56:23
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''


import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from common.memory import ReplayBuffer
from common.model import MLP2
class DQN:
    def __init__(self, n_states, n_actions, cfg):
        
        self.n_actions = n_actions  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma # 奖励的折扣因子
        # e-greedy策略相关参数
        self.sample_count = 0 # 用于epsilon的衰减计数, 保存训练步数，每多少步之后开始衰减
        self.epsilon = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.policy_net = MLP2(n_states, n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP2(n_states, n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        # target_net的初始模型参数完全复制policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 不启用 BatchNormalization 和 Dropout
        # 可查parameters()与state_dict()的区别，前者require_grad=True
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss = 0
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state, train=True):
        """
        选择动作
        :param state: eg: [ 0.03073904  0.00145001 -0.03088818 -0.03131252]
        :param train: 是否训练模式
        :return:
        """
        if train:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * self.sample_count / self.epsilon_decay)
            self.sample_count += 1
            # 保持随机探索或最优动作
            if random.random() > self.epsilon:
                with torch.no_grad():
                    # 先转为张量便于丢给神经网络,state元素数据原本为float64
                    # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价, 形状变成 torch.Size([1, 4])
                    state = torch.tensor(
                        [state], device=self.device, dtype=torch.float32)
                    # 开始前向网络计算，输出每个action的logits， 如tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)
                    q_value = self.policy_net(state)
                    # tensor.max(1)返回每行的最大值以及对应的下标，
                    # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                    # 所以tensor.max(1)[1]返回最大值对应的下标，即action, 得到一个action
                    action = q_value.max(1)[1].item()  
            else:
                action = random.randrange(self.n_actions)
            return action
        else: 
            with torch.no_grad(): # 取消保存梯度
                    # 先转为张量便于丢给神经网络,state元素数据原本为float64
                    # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                    state = torch.tensor(
                        [state], device='cpu', dtype=torch.float32) # 如tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)
                    q_value = self.target_net(state)
                    # tensor.max(1)返回每行的最大值以及对应的下标，
                    # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                    # 所以tensor.max(1)[1]返回最大值对应的下标，即action
                    action = q_value.max(1)[1].item() 
            return action
    def update(self):
        # 如果小于一个batch_size的时候就不会更新
        if len(self.memory) < self.batch_size:
            return
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        '''转为张量, 形状为 [batch_size, state_num]
        例如tensor([[-4.5543e-02, -2.3910e-01,  1.8344e-02,  2.3158e-01],...,[-1.8615e-02, -2.3921e-01, -1.1791e-02,  2.3400e-01]])'''
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  # 例如tensor([[1],...,[0]]), shape: [batch_size, 1], 选择的只有一个动作
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1]), shape (batch_size,)
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)   #shape [batch_size, state_num]
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device).unsqueeze(1)  # 将bool转为float然后转为张量, shape (batch_size,1)

        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        '''torch.gather:对于a=torch.Tensor([[1,2],[3,4]]),那么a.gather(1,torch.Tensor([[0],[1]]))=torch.Tensor([[1],[3]])'''
        q_values0 = self.policy_net(state_batch) # 前向计算, shape [batch_size, action_num],   eg:torch.Size([64, 2])
        q_values = q_values0.gather(dim=1, index=action_batch)  # index表示当采取这个action时，这个对应的位置的采取的action的概率
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states
        next_q_value = self.target_net(next_state_batch)  # 目标网络获取的下一个state对应的action值的概率, [batch_size, action_num],
        next_state_values = next_q_value.max(1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,]), (batch_size,) 获取最大的那个action的概率
        # 计算 expected_q_value， 核心计算Q值公式，奖励+折扣因子*max(), done_batch[0]是表示是否结束了，这个batch的第一个元素
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward， reward_batch, next_state_values,（batch_size,)
        expected_q_values = reward_batch + self.gamma * \
            next_state_values * (1-done_batch[0])  # 得到的形状 shape (batch_size,)
        # self.loss = F.smooth_l1_loss(q_values,expected_q_values.unsqueeze(1)) # 计算 Huber loss
        self.loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算policy网络和target网络的价值函数的不同，计算损失 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()  # 根据梯度更新网络参数

    def save(self,path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self,path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))  
