# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:58:08 2022

@author: Tuan
"""

import numpy as np


memory_size = int(1e7) # maximum size of buffer memory

n_episodes = int(1e4) # number of simulation episodes
n_steps = int(1e4) # number of steps per episode

epsilon = 1.
epsilon_decay = 1e-4 # noise decay rate

polyak = 0.01 # target network soft update rate

gamma = 0.99 # reward discount factor

actor_lr = 1e-2 # learning rate of actor network
critic_lr = 1e-2 # learning rate of critic network

hidden1_dim = 512
hidden2_dim = 256

checkpoint_path = './checkpoints/MADDPG_PER/'

n_uavs = 3
n_antens = 20
Nx = 4
Ny = 5
Nrf = 4
Emax = 1e5
Hrange = (100, 150)
Vmax = 10
r_buf = 50

Xrange = (0, 200)
Yrange = (0, 200)

n_users = 15

noise = 1e-11
power = 1e2
frequency = 38e9
c = 299792458

U_tip = 120 # tip spd of rotor blade (m/s), = Omega * R
v_0 = 4.03 # mean rotor induced velocity in hover, = sqrt(W/2*rho*A)
d_0 = 0.6 # fuselage drag radtio
rho = 1.225 # air density (kg/m^3)
s = 0.05 # rotor solidity
A = 0.503 # rotor disc area (m^2), = pi*R^2
delta = 0.012 # profile drag coefficient
Omega = 300 # blade angular velocity (rad/s)
R = 0.4 # rotor radius (m)
k = 0.1 # incremental correction factor to induced power
W = 20 # aircraft weight (N)
P_b = delta/8 * rho * s * A * Omega**3 * R**3
P_i = (1 + k) * np.sqrt(W**3)/np.sqrt(2 * rho * A)

C_1, C_2, zeta_LoS, zeta_NLoS = (9.61, 0.16, 1, 20)