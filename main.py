
import numpy as np
import configs as cf
from environment import Environment
from agent import MADDPG_Agents


obs_dim = 3 + 1 + 1 + 1 + 2 * cf.Nrf
action_dim = 1 + 1 + 1 + cf.n_antens * cf.Nrf * 2 + cf.Nrf * cf.Nrf * 2


ev = Environment()
agents = MADDPG_Agents(obs_dim, action_dim)

s0 = ev.reset()
a0 = agents.get_all_actions(s0)

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
    
spd, azi, ele, analog_bf, digital_bf = process_action(a0)

sumrate, step_reward, next_obs, done = ev.step(spd, azi, ele, analog_bf, digital_bf)

agents.memory.push(s0, a0, step_reward, next_obs, done)

(state_batch, action_batch, reward_batch, next_state_batch, done_batch),\
batch_dx, IS_weights = agents.memory.sample(5)

actor_loss, critic_loss = agents.learn()

print(actor_loss)
print(critic_loss)

# state_batch, \
# action_batch, \
# reward_batch, \
# next_state_batch, \
# done_batch = agents.memory.sample(3)