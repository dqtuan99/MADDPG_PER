
import numpy as np
import configs as cf
import matplotlib.pyplot as plt
from environment import Environment
from agent import MADDPG_Agents
from tqdm import tqdm


obs_dim = 3 + 1 + 1 + 1 + 2 * cf.Nrf
action_dim = 1 + 1 + 1 + cf.n_antens * cf.Nrf * 2 + cf.Nrf * cf.Nrf * 2


ev = Environment()
agents = MADDPG_Agents(obs_dim, action_dim)

# s0 = ev.reset()
# a0 = agents.get_all_actions(s0)

def plot_uavs_trajectory(uavs_trajectory, users_position, title):
    '''
    '''
    ax = plt.axes(projection='3d')
    uavs_colors = ['red', 'green', 'blue']
    uavs_start_markers = ['r', 'g', 'b']
    
    for m in range(uavs_trajectory.shape[0]):
        xline = uavs_trajectory[m].T[0]
        yline = uavs_trajectory[m].T[1]
        zline = uavs_trajectory[m].T[2]
        ax.plot3D(xline, yline, zline, uavs_colors[m])
        ax.plot([uavs_trajectory[m,0,0]], [uavs_trajectory[m,0,1]], [uavs_trajectory[m,0,2]], markerfacecolor=uavs_start_markers[m], markeredgecolor='k', marker='o', markersize=10, alpha=1)
        ax.plot([uavs_trajectory[m,-1,0]], [uavs_trajectory[m,-1,1]], [uavs_trajectory[m,-1,2]], markerfacecolor=uavs_start_markers[m], markeredgecolor='k', marker='X', markersize=10, alpha=1)
    
    
    for user in users_position:
        ax.plot([user[0]], [user[1]], [user[2]], markerfacecolor='r', markeredgecolor='k', marker='o', markersize=8, alpha=0.6)
    
    # plot charging destination
    edp = np.append(ev.ending_point, 0)
    ax.plot([edp[0]], [edp[1]], [edp[2]], markerfacecolor='y', markeredgecolor='k', marker='o', markersize=15, alpha=0.6)
    
    plt.title(title)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_zlabel('height')
    plt.show()

def plot_learning(plot_reward, xlabel, ylabel, title):
    temp_plot = np.array(plot_reward)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(temp_plot)
    plt.title(title)
    plt.show()

def process_action(action):
    # spd = np.zeros(cf.n_uavs)
    # azi = np.zeros(cf.n_uavs)
    # ele = np.zeros(cf.n_uavs)
    # analog_bf = np.zeros_like(ev.analog_bf)
    # digital_bf = np.zero_like(ev.digital_bf)
    
    spd = action[:, 0] * cf.Vmax
    azi = action[:, 1] * 2 * np.pi
    ele = action[:, 2] * 2 * np.pi - np.pi
    
    analog_bf = action[:, 3 : 3 + cf.n_antens * cf.Nrf * 2]
    analog_bf = (analog_bf[:, ::2] + analog_bf[:, 1::2] * 1j).reshape((cf.n_uavs, cf.n_antens, cf.Nrf))
    
    digital_bf = action[:, 3 + cf.n_antens * cf.Nrf * 2 : ]
    digital_bf = (digital_bf[:, ::2] + digital_bf[:, 1::2] * 1j).reshape((cf.n_uavs, cf.Nrf, cf.Nrf))
    
    return spd, azi, ele, analog_bf, digital_bf


n_episodes = cf.n_episodes
n_steps = cf.n_steps
min_buffer = 50
print_interval = 20

ep_sumrate_hist = []
ep_reward_hist = []
actor_loss_hist = []
critic_loss_hist = []

best_reward = -np.Inf

for episode in range(n_episodes):
    obs = ev.reset()
    agents.noise.reset()
    agents.epsilon = cf.epsilon
    ep_reward = 0.
    ep_sumrate = 0.
    
    for step in tqdm(range(n_steps), desc='Episode ' + str(episode) + ' progress bar: '):
        action = agents.get_all_actions(obs)
        spd, azi, ele, analog_bf, digital_bf = process_action(action)
        
        sumrate, step_reward, next_obs, done, other_rewards = ev.step(spd, azi, ele, analog_bf, digital_bf)
        
        ep_reward += step_reward.mean()
        ep_sumrate += sumrate
        
        agents.memory.store_transition(obs, action, step_reward, next_obs, done)
        
        if agents.memory.mem_ptr > min_buffer:
            actor_loss, critic_loss = agents.learn()
        
        obs = next_obs        
        
        if done.all() == True:
            break
        
    
    ep_reward /= (step + 1)
    ep_reward_hist.append(ep_reward)
    avg_reward_last_100_ep = np.mean(ep_reward_hist[-100:])
    
    ep_sumrate /= (step + 1)
    ep_sumrate_hist.append(ep_sumrate)
    avg_sumrate_last_100_ep = np.mean(ep_sumrate_hist[-100:])
    
    actor_loss_hist.append(actor_loss)
    critic_loss_hist.append(critic_loss)
    
    print('ep', episode, '| ep_reward =', ep_reward, '| avg_reward_last_100_ep =', avg_reward_last_100_ep)
    print('ep_sumrate =', ep_sumrate, '| avg_sumrate_last_100_ep =', avg_sumrate_last_100_ep)
    
    if episode % print_interval == (print_interval - 1):
        plot_learning(ep_reward_hist, 'Episode', 'Avg Episode Reward', 'Reward per Episode')
        plot_learning(ep_sumrate_hist, 'Episode', 'Avg Sumrate', 'Reward per Episode')
        plot_learning(actor_loss_hist, 'Episode', 'Actor Loss', 'Actor Loss per Episode')
        plot_learning(critic_loss_hist, 'Episode', 'Critic Loss', 'Critic Loss per Episode')
    
    np.savetxt('logs/ep_reward.csv', ep_reward_hist, delimiter=',')
    np.savetxt('logs/ep_sumrate.csv', ep_reward_hist, delimiter=',')
    np.savetxt('logs/actor_loss.csv', actor_loss_hist, delimiter=',')
    np.savetxt('logs/critic_loss.csv', critic_loss_hist, delimiter=',')
    
    uavs_trajectory = ev.uavs_trajectory.reshape((cf.n_uavs, -1, 3))[:, 1:]
    plot_uavs_trajectory(uavs_trajectory, ev.users_pos, 'UAVs trajectory of ep ' + str(episode))    
    
    np.savetxt('logs/trajectory/trajectory_ep_' + str(episode) + '.csv', uavs_trajectory.flatten(), delimiter=',')
    
    if ep_reward > best_reward:
        best_reward = ep_reward
        agents.save_models()
        
    
# spd, azi, ele, analog_bf, digital_bf = process_action(a0)

# sumrate, step_reward, next_obs, done, other_rewards = ev.step(spd, azi, ele, analog_bf, digital_bf)

# agents.memory.push(s0, a0, step_reward, next_obs, done)

# (state_batch, action_batch, reward_batch, next_state_batch, done_batch),\
# batch_dx, IS_weights = agents.memory.sample(5)

# actor_loss, critic_loss = agents.learn()

# print(actor_loss)
# print(critic_loss)

# state_batch, \
# action_batch, \
# reward_batch, \
# next_state_batch, \
# done_batch = agents.memory.sample(3)