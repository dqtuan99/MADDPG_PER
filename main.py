
import numpy as np
import configs as cf
import matplotlib.pyplot as plt
from environment import Environment
from agent import MADDPG_Agents
from tqdm import tqdm


ev = Environment()
agents = MADDPG_Agents(ev.obs_dim, ev.action_dim)

n_episodes = cf.n_episodes
n_steps = cf.n_steps
min_buffer = 50
print_interval = 1

best_reward = -np.Inf

def plot_uavs_trajectory(uavs_trajectory, users_position, title):
    '''
    '''
    ax = plt.axes(projection='3d')
    uavs_colors = ['blue', 'red', 'green']
    uavs_start_markers = ['b', 'r', 'g']
    
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
    edp = np.append(ev.destination, 0)
    ax.plot([edp[0]], [edp[1]], [edp[2]], markerfacecolor='y', markeredgecolor='k', marker='o', markersize=15, alpha=0.6)
    
    plt.title(title)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_zlabel('height')
    plt.savefig('./figs/trajectories/' + title + '.png', dpi=144)
    plt.show()

def plot_learning(plot_reward, xlabel, ylabel, title):
    temp_plot = np.array(plot_reward)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(temp_plot)
    plt.title(title)
    plt.savefig('./figs/results/' + title + '.png', dpi=144)
    plt.show()

# def process_action(action):
#     # spd = np.zeros(cf.n_uavs)
#     # azi = np.zeros(cf.n_uavs)
#     # ele = np.zeros(cf.n_uavs)
#     # analog_bf = np.zeros_like(ev.analog_bf)
#     # digital_bf = np.zero_like(ev.digital_bf)
    
#     spd = action[:, 0] * cf.Vmax
#     azi = action[:, 1] * 2 * np.pi
#     ele = action[:, 2] * 2 * np.pi - np.pi
    
#     analog_bf = action[:, 3 : 3 + cf.n_antens * cf.Nrf * 2]
#     analog_bf = (analog_bf[:, ::2] + analog_bf[:, 1::2] * 1j).reshape((cf.n_uavs, cf.n_antens, cf.Nrf))
    
#     digital_bf = action[:, 3 + cf.n_antens * cf.Nrf * 2 : ]
#     digital_bf = (digital_bf[:, ::2] + digital_bf[:, 1::2] * 1j).reshape((cf.n_uavs, cf.Nrf, cf.Nrf))
    
#     return spd, azi, ele, analog_bf, digital_bf


def process_action(action):
    spd = action[:, 0] * cf.Vmax
    azi = action[:, 1] * 2 * np.pi
    ele = action[:, 2] * 2 * np.pi - np.pi
    association = action[:, 3:]
    association = np.argpartition(association, -ev.Nrf)[:, -ev.Nrf:]
    
    return spd, azi, ele, association


ep_reward_hist = []
ep_reward_hist2 = []

ep_sumrate_hist = []
ep_sumrate_hist2 = []

actor_loss_hist = []
critic_loss_hist = []

remaining_energy_hist = []
d_destination_hist = []

dest_reward_hist = []
step_penalty_hist = []
collision_penalty_hist = []
OOB_penalty_hist = []

for episode in range(n_episodes):
    obs = ev.reset()
    agents.noise.reset()
    agents.epsilon = cf.epsilon
    
    ep_reward = 0.0
    ep_sumrate = 0.0
    ep_dest_reward = 0.0
    ep_step_penalty = 0.0
    ep_collision_penalty = 0.0
    ep_OOB_penalty = 0.0
    
    reward_list = []
    sumrate_list = []
    dest_reward_list = []
    step_penalty_list = []
    collision_penalty_list = []
    OOB_penalty_list = []
    
    for step in tqdm(range(n_steps), desc='Episode ' + str(episode + 1) + ': '):
        
        action = agents.get_all_actions(obs)
        
        # spd, azi, ele, analog_bf, digital_bf = process_action(action)
        spd, azi, ele, association = process_action(action)
        
        sumrate, step_reward, next_obs, penalties = ev.step(spd, azi, ele, association)
        
        dest_reward, step_penalty, collision_penalty, OOB_penalty = penalties
        
        ep_reward += step_reward.sum()
        ep_sumrate += sumrate.sum()
        ep_dest_reward += dest_reward.sum()
        ep_step_penalty += step_penalty.sum()
        ep_collision_penalty += collision_penalty.sum()
        ep_OOB_penalty += OOB_penalty.sum()
        
        reward_list.append(step_reward)
        sumrate_list.append(sumrate)
        dest_reward_list.append(dest_reward)
        step_penalty_list.append(step_penalty)
        collision_penalty_list.append(collision_penalty)
        OOB_penalty_list.append(OOB_penalty)
        
        agents.memory.store_transition(obs, action, step_reward, next_obs)
        
        if agents.memory.mem_ptr > min_buffer:
            actor_loss, critic_loss = agents.learn()
        
        obs = next_obs
        
    ep_reward_hist.append(ep_reward)
    avg_reward_last_100_ep = np.mean(ep_reward_hist[-100:])
    ep_reward_hist2.append(avg_reward_last_100_ep)
    
    ep_sumrate_hist.append(ep_sumrate)
    avg_sumrate_last_100_ep = np.mean(ep_sumrate_hist[-100:])
    ep_sumrate_hist2.append(avg_sumrate_last_100_ep)
    
    actor_loss_hist.append(np.array(actor_loss))
    critic_loss_hist.append(np.array(critic_loss))
    
    remaining_energy_hist.append(ev.uavs_battery)
    d_destination_hist.append(ev.d_destination)

    dest_reward_hist.append(ep_dest_reward)
    step_penalty_hist.append(ep_step_penalty)
    collision_penalty_hist.append(ep_collision_penalty)
    OOB_penalty_hist.append(ep_OOB_penalty)
    
    print('=============================')
    print('Episode', episode + 1)
    print('Average remaining energy =', ev.uavs_battery.mean())
    print('Average remaining distance to destination =', ev.d_destination.mean())
    print('Ep Reward =', ep_reward)
    print('Ep Sumrate =', ep_sumrate)
    print('Ep Destination Encourage Rewards =', ep_dest_reward)
    print('Ep Step Penalty =', ep_step_penalty)
    print('Ep Collision Penalty =', ep_collision_penalty)
    print('Ep OOB Penalty =', ep_OOB_penalty)
    print('Avg Reward Last 100 Ep =', avg_reward_last_100_ep)
    print('Avg Sumrate Last 100 Ep =', avg_sumrate_last_100_ep)
    print('=============================')
    
    np.savetxt('logs/ep_reward.csv', ep_reward_hist, delimiter=',')
    np.savetxt('logs/ep_reward2.csv', ep_reward_hist2, delimiter=',')
    np.savetxt('logs/ep_sumrate.csv', ep_sumrate_hist, delimiter=',')
    np.savetxt('logs/ep_sumrate2.csv', ep_sumrate_hist2, delimiter=',')
    
    np.savetxt('logs/actor_loss.csv', actor_loss_hist, delimiter=',')
    np.savetxt('logs/critic_loss.csv', critic_loss_hist, delimiter=',')
    
    np.savetxt('logs/remain_energy.csv', remaining_energy_hist, delimiter=',')
    np.savetxt('logs/destination_distance.csv', d_destination_hist, delimiter=',')
    
    np.savetxt('logs/ep_dest_reward.csv', dest_reward_hist, delimiter=',')
    np.savetxt('logs/ep_step_pen.csv', step_penalty_hist, delimiter=',')
    np.savetxt('logs/ep_col_pen.csv', collision_penalty_hist, delimiter=',')
    np.savetxt('logs/ep_oob_pen.csv', OOB_penalty_hist, delimiter=',')
    
    uavs_trajectory = ev.uavs_trajectory.reshape((cf.n_uavs, -1, 3))[:, 1:]
    # plot_uavs_trajectory(uavs_trajectory, ev.users_pos, 'UAVs trajectory of ep ' + str(episode))    
    
    np.savetxt('logs/trajectory/trajectory_ep_' + str(episode) + '.csv', uavs_trajectory.flatten(), delimiter=',')
    
    if episode % print_interval == (print_interval - 1):
        plot_learning(ep_reward_hist, 'Episode', 'Reward', 'Reward per Episode')
        plot_learning(ep_reward_hist2, 'Episode', 'Reward', 'Reward per Episode (2)')
        
        plot_learning(ep_sumrate_hist, 'Episode', 'Sumrate', 'Sumrate per Episode')
        plot_learning(ep_sumrate_hist2, 'Episode', 'Sumrate', 'Sumrate per Episode (2)')
        
        plot_learning(actor_loss_hist, 'Episode', 'Actor Loss', 'Actor Loss per Episode')
        plot_learning(critic_loss_hist, 'Episode', 'Critic Loss', 'Critic Loss per Episode')
        
                
        plot_learning(reward_list, 'Step', 'Reward', 'Reward per Step of Ep ' + str(episode + 1))
        plot_learning(sumrate_list, 'Step', 'Sumrate', 'Sumrate per Step of Ep ' + str(episode + 1))
        plot_learning(dest_reward_list, 'Step', 'Reward', 'Destination Encourage Reward per Step of Ep ' + str(episode + 1))
        plot_learning(step_penalty_list, 'Step', 'Penalty', 'Step penalty per Step of Ep ' + str(episode + 1))
        plot_learning(collision_penalty_list, 'Step', 'Penalty', 'Collision penalty per Step of Ep ' + str(episode + 1))
        plot_learning(OOB_penalty_list, 'Step', 'Penalty', 'OOB penalty per Step of Ep ' + str(episode + 1))
        
    plot_uavs_trajectory(uavs_trajectory, ev.users_pos, 'UAVs trajectory of ep ' + str(episode + 1)) 
    
    if ep_reward > best_reward:
        best_reward = ep_reward
        agents.save_models()
        
        
        