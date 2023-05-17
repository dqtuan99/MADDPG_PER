
import numpy as np
import torch
import torch.nn.functional as F
import configs as cf
from buffer import ReplayBuffer
from noise import OUActionNoise
from networks import ActorNet, CriticNet


def soft_update(main_net, target_net):
    for main_param, target_param in zip(main_net.parameters(), target_net.parameters()):
        target_param.data.copy_(cf.polyak * target_param.data + (1 - cf.polyak) * main_param.data)


class MADDPG_Agents():
    def __init__(self, obs_dim, action_dim):
        
        super(MADDPG_Agents, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.n_agents = cf.n_agents        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.gamma = cf.gamma
        
        self.batch_size = cf.batch_size
        
        self.memory = ReplayBuffer(cf.memory_size, cf.n_agents, obs_dim, action_dim)
        
        self.noise = OUActionNoise(np.zeros(action_dim))
        
        self.actors = [ActorNet(obs_dim, action_dim, 'actor_' + str(agent_idx)) 
                       for agent_idx in range(self.n_agents)]
        
        self.critics = [CriticNet(obs_dim, action_dim, 'critic_' + str(agent_idx))
                        for agent_idx in range(self.n_agents)]
        
        self.target_actors = [ActorNet(obs_dim, action_dim, 'target_actor_' + str(agent_idx)) 
                              for agent_idx in range(self.n_agents)]
        
        self.target_critics = [CriticNet(obs_dim, action_dim, 'target_critic_' + str(agent_idx))
                               for agent_idx in range(self.n_agents)]
        
        # self.polyak = cf.polyak
        
        self.epsilon = cf.epsilon
        
    
    def get_all_actions(self, all_obs, is_training=True):
        all_actions = []
        
        for agent_idx in range(self.n_agents):
            obs = torch.FloatTensor(all_obs[agent_idx]).to(self.device)
            action = self.actors[agent_idx](obs).cpu()
            
            if is_training:
                noise = torch.FloatTensor(self.noise())
                action += noise * self.epsilon
                self.epsilon = max(0, self.epsilon - cf.epsilon_decay)
                all_actions.append(action.cpu().detach().numpy())
        
        all_actions = np.array(all_actions)
        
        return all_actions
    
    
    def learn(self):
        all_actor_loss = []
        all_critic_loss = []
        
        # (all_obs_batch, all_action_batch, all_reward_batch, all_next_obs_batch, all_done_batch),\
        # all_batch_idx, all_IS_weights = self.memory.sample(self.batch_size)
        
        # all_obs_batch, all_action_batch, all_reward_batch, \
        # all_next_obs_batch, _ = self.memory.sample_buffer(self.batch_size)
        
        all_obs_batch, all_action_batch, all_reward_batch, \
                all_next_obs_batch = self.memory.sample_buffer(self.batch_size)
        
        all_obs_batch = torch.FloatTensor(all_obs_batch).to(self.device)
        all_action_batch = torch.FloatTensor(all_action_batch).to(self.device)
        all_reward_batch = torch.FloatTensor(all_reward_batch).to(self.device)
        # all_reward_batch = torch.einsum('abc->acb', all_reward_batch)
        all_next_obs_batch = torch.FloatTensor(all_next_obs_batch).to(self.device)
        # all_done_batch = torch.BoolTensor(all_done_batch).to(self.device)
    
        joint_obs_batch = all_obs_batch.reshape((self.batch_size, -1))
        joint_action_batch = all_action_batch.reshape((self.batch_size, -1))
        joint_next_obs_batch = all_next_obs_batch.reshape((self.batch_size, -1))        
                    
        for agent_idx in range(self.n_agents):
            reward_batch = all_reward_batch[:, agent_idx].unsqueeze(1)
            # done_batch = all_done_batch[:, agent_idx].reshape((-1, 1))
            
            self.critics[agent_idx].optimizer.zero_grad()
            
            current_Q = self.critics[agent_idx](joint_obs_batch, joint_action_batch)
            
            joint_target_mu_batch = torch.stack([self.target_actors[m](all_next_obs_batch[:, m, :]) 
                                           for m in range(self.n_agents)])
            joint_target_mu_batch = torch.einsum('abc->bac', joint_target_mu_batch).reshape((self.batch_size, -1))
            
            target_Q = self.target_critics[agent_idx](joint_next_obs_batch, joint_target_mu_batch)
            
            # y = reward_batch + self.gamma * ~done_batch * target_Q
            y = reward_batch + self.gamma * target_Q
                                    
            critic_loss = F.mse_loss(current_Q, y)
            critic_loss.backward()
            
            self.critics[agent_idx].optimizer.step()            
                        
            self.actors[agent_idx].optimizer.zero_grad()            
            
            obs_batch = all_obs_batch[:, agent_idx, :]
            mu_batch = self.actors[agent_idx](obs_batch)
            temp = all_action_batch.clone()
            temp[:, agent_idx, :] = mu_batch
            joint_mu_batch = temp.view((self.batch_size, -1))
            
            actor_loss = -self.critics[agent_idx](joint_obs_batch, joint_mu_batch)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            
            self.actors[agent_idx].optimizer.step()
            
            all_actor_loss.append(actor_loss.detach())
            all_critic_loss.append(critic_loss.detach())
            
        
        for agent_idx in range(self.n_agents):
            soft_update(self.actors[agent_idx], self.target_actors[agent_idx])
            soft_update(self.critics[agent_idx], self.target_critics[agent_idx])
        
                    
        # return sum(all_actor_loss).item()/self.n_agents, sum(all_critic_loss).item()/self.n_agents
        return all_actor_loss, all_critic_loss
        
    def save_models(self):
        for agent_idx in range(self.n_agents):
            self.actors[agent_idx].save_checkpoint()
            self.critics[agent_idx].save_checkpoint()
            self.target_actors[agent_idx].save_checkpoint()
            self.target_critics[agent_idx].save_checkpoint()
            
        # joint_mu = []
        # joint_target_mu_batch = []
        # for agent_idx in range(self.n_agents):            
        #     obs_batch = torch.FloatTensor(all_obs_batch[:, agent_idx, :]).to(self.device)
        #     next_obs_batch = torch.FloatTensor(all_next_obs_batch[:, agent_idx, :]).to(self.device)
            
        #     mu = self.actors[agent_idx](obs_batch).detach().numpy()
        #     target_mu = self.target_actors[agent_idx](next_obs_batch).detach().numpy()
            
        #     joint_mu.append(mu)
        #     joint_target_mu_batch.append(target_mu)
        
        # joint_mu = np.array(joint_mu)
        # joint_target_mu_batch = np.array(joint_mu)
        
        # joint_mu = np.einsum('abc->bac', joint_mu)
        # joint_target_mu_batch = np.einsum('abc->bac', joint_target_mu_batch)        
        
        # joint_mu = torch.FloatTensor(joint_mu).reshape((self.batch_size, -1)).to(self.device)
        # joint_target_mu_batch = torch.FloatTensor(joint_target_mu_batch).reshape((self.batch_size, -1)).to(self.device)
        
        # self.joint_mu = joint_mu
        # self.joint_obs_batch = joint_obs_batch
        
        # for agent_idx in range(self.n_agents):
        #     obs_batch = all_obs_batch[:, agent_idx, :]
        #     action_batch = all_action_batch[:, agent_idx, :]
        #     reward_batch = all_reward_batch[:, agent_idx, :]
        #     next_obs_batch = all_next_obs_batch[:, agent_idx, :]
            
        #     self.actors[agent_idx].optimizer.zero_grad()
            
        #     actor_loss = - self.critics[agent_idx](joint_obs_batch, joint_mu).mean()
        #     actor_loss.backward()
            
        #     self.actors[agent_idx].optimizer.step()
        
        #     self.critics[agent_idx].optimizer.zero_grad()
            
        #     targets = reward_batch + self.gamma * self.target_critics[agent_idx](
        #         joint_next_obs_batch, joint_target_mu_batch)
            
        #     critics = self.critics[agent_idx](joint_obs_batch, joint_action_batch)
            
        #     critic_loss = F.mse_loss(targets, critics) / self.batch_size
            
        #     critic_loss.backward()
        #     self.critics[agent_idx].optimizer.step()
            
        #     all_actor_loss.append(actor_loss)
        #     all_critic_loss.append(critic_loss)
            
        #     soft_update(self.actors[agent_idx], self.target_actors[agent_idx])
        #     soft_update(self.critics[agent_idx], self.target_critics[agent_idx])
        
        # return sum(actor_loss).item()/self.n_agents, sum(critic_loss).item()/self.n_agents
        
        
        