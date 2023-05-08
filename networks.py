
import os
import torch
import torch.nn as nn
import torch.optim as optim

import configs as cf


class ActorNet(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 name):
        
        super(ActorNet, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.checkpoint = os.path.join(cf.checkpoint_path, name + '.csv')
        
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, cf.hidden1_dim),
            nn.LayerNorm(cf.hidden1_dim),
            nn.ReLU(),
            nn.Linear(cf.hidden1_dim, cf.hidden2_dim),
            nn.LayerNorm(cf.hidden2_dim),
            nn.ReLU(),
            nn.Linear(cf.hidden2_dim, action_dim),
            nn.Sigmoid()
            ).to(cf.device)
        
        nn.init.kaiming_uniform_(self.policy[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.policy[0].weight)
        nn.init.kaiming_uniform_(self.policy[3].weight, nonlinearity='relu')
        nn.init.zeros_(self.policy[3].weight)
        nn.init.xavier_uniform_(self.policy[6].weight)
        nn.init.zeros_(self.policy[6].weight)
        
        self.optimizer = optim.Adam(self.parameters(), lr=cf.actor_lr)
        
    def forward(self, state):
        return self.policy(state)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))
        

class CriticNet(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 name):
        
        super(CriticNet, self).__init__()
        
        self.n_agents = cf.n_agents
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.checkpoint = os.path.join(cf.checkpoint_path, name + '.csv')
        
        self.obs_encoding = nn.Sequential(
            nn.Linear(self.n_agents * obs_dim, cf.hidden1_dim),
            nn.LayerNorm(cf.hidden1_dim),
            nn.ReLU()
            ).to(cf.device)
                
        self.actions_encoding = nn.Sequential(
            nn.Linear(self.n_agents * action_dim, cf.hidden1_dim),
            nn.LayerNorm(cf.hidden1_dim),
            nn.ReLU()
            ).to(cf.device)
        
        self.critic = nn.Sequential(
            nn.Linear(cf.hidden1_dim + cf.hidden1_dim, cf.hidden1_dim),
            nn.LayerNorm(cf.hidden1_dim),
            nn.ReLU(),
            nn.Linear(cf.hidden1_dim, cf.hidden2_dim),
            nn.LayerNorm(cf.hidden2_dim),
            nn.ReLU(),
            nn.Linear(cf.hidden2_dim, 1)
            ).to(cf.device)
        
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
        
        self.optimizer = optim.Adam(self.parameters(), lr=cf.actor_lr)
        
    def forward(self, state, actions):
        state = self.obs_encoding(state)
        actions = self.actions_encoding(actions)
        
        return self.critic(torch.hstack([state, actions]))
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint))
       