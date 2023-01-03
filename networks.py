# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:16:46 2022

@author: Tuan
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from settings import checkpoint_path


class ActorNet(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden1_dim,
                 hidden2_dim,
                 actor_lr,
                 device,
                 name):
        
        super(ActorNet, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.checkpoint = os.path.join(checkpoint_path, name)
        
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LayerNorm(hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, action_dim),
            nn.sigmoid()
            ).to(device)
        
        nn.init.kaiming_uniform_(self.policy[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.policy[0].weight)
        nn.init.kaiming_uniform_(self.policy[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.policy[3].weight)
        nn.init.xavier_uniform_(self.policy[6].weight)
        nn.init.zeros_(self.policy[6].weight)
        
    def forward(self, state):
        return self.policy(state)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))
        

class CriticNet(nn.Module):
    def __init__(self,
                 n_agents,
                 obs_dim,
                 action_dim,
                 hidden1_dim,
                 hidden2_dim,
                 critic_lr,
                 device,
                 name,
                 checkpoint_path
                 ):
        
        super(CriticNet, self).__init__()
        
        self.n_agents = n_agents
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.checkpoint = os.path.join(checkpoint_path, name)
        
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        
        self.obs_encoding = nn.Sequential(
            nn.Linear(obs_dim, hidden1_dim),
            nn.LayerNorm(),
            nn.ReLU()
            ).to(device)
                
        self.actions_encoding = nn.Sequential(
            nn.Linear(n_agents * action_dim, hidden1_dim),
            nn.LayerNorm(),
            nn.ReLU()
            ).to(device)
        
        self.critic = nn.Sequential(
            nn.Lieanr(hidden1_dim + hidden1_dim, hidden1_dim),
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Lieanr(hidden1_dim, hidden2_dim),
            nn.LayerNorm(),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1)
            ).to(device)
        
        nn.init.kaiming_uniform_(self.obs_encoding[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.obs_encoding[0].bias)
        nn.init.kaiming_uniform_(self.actions_encoding[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.actions_encoding[0].bias)
        nn.init.kaiming_uniform_(self.critic[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.critic[0].bias)
        nn.init.kaiming_uniform_(self.critic[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.critic[3].bias)
        nn.init.xavier_uniform_(self.critic[6].weight)
        nn.init.zeros_(self.critic[6].bias)
        
    def forward(self, state, actions):
        state = self.obs_encoding(state)
        actions = self.actions_encoding(actions)
        
        return self.critic(torch.cat([state, actions]))
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))
       