# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:49:35 2022

@author: TUAN
"""

import random
import numpy as np
import torch
from settings import *
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances
from ortools.linear_solver import pywraplp


def const_modulus(cmplx_num, const_value):
    modulus = np.abs(cmplx_num)
    n = modulus/const_value
    
    return cmplx_num/n


class Environment():
    def __init__(self):
        
        self.Xmin, self.Xmax = Xrange
        self.Ymin, self.Ymax = Yrange
        
        self.n_uavs = n_uavs
        
        self.n_antens = n_antens
        self.Nx = Nx
        self.Ny = Ny
        self.Nrf = Nrf
        self.power = power
        
        self.Hmin, self.Hmax = Hrange
        self.Vmax = Vmax
        self.Emax = Emax
        self.r_buf = r_buf
    
        self.n_users = n_users
        
        self.noise = noise
        self.lambda_ = c/frequency
                
        # self.uavs_pos = np.zeros((self.n_uavs, 3))
        # self.users_pos = np.zeros((self.n_users, 3))
        # self.uavs_battery = np.array([self.Emax] * self.n_uavs)
        self.ending_point = np.array([0, 0])
        
        self.obs_dim = 3 + 1 + 1 + 2*self.Nrf + \
            2*self.n_antens*self.Nrf + 2*self.Nrf*self.Nrf
            
        self.action_dim = 1 + 1 + 1 + 2*self.n_antens*self.Nrf + 2*self.Nrf*self.Nrf
        
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        
        self.reset()
    
    
    def update_uavs_pos(self, V, azi, ele):
        dx = V * np.cos(azi) * np.cos(ele) * self.dt
        dy = V * np.sin(azi) * np.cos(ele) * self.dt
        dz = V * np.sin(ele) * self.dt
        
        self.uavs_pos[:, 0] += dx
        self.uavs_pos[:, 1] += dy
        self.uavs_pos[:, 2] += dz
        
        self.uavs_pos[:, 2][np.where(self.uavs_pos[:, 2] < self.Hmin)] = self.Hmin
        self.uavs_pos[:, 2][np.where(self.uavs_pos[:, 2] > self.Hmax)] = self.Hmax
        
        return self.uavs_pos
        
    
    def update_uavs_battery(self, V):        
        cost = P_b * (1 + 3 * V**2/U_tip**2) + \
               P_i * np.sqrt( np.sqrt(1 + V**4/(4*v_0**4)) - V**2/(2*v_0**2) ) + \
               d_0 * rho * s * A * V**3/2
        
        self.uavs_battery -= cost*self.dt
        
        return self.uavs_battery
    
    
    def uavs_move(self, V, azi, ele):
        self.uavs_pos = self.update_uavs_pos(V, azi, ele)
        self.uavs_battery = self.update_uavs_battery(V)
        
        return self.uavs_pos, self.uavs_battery
    
    
    def get_nearby_uav_dist(self):
        dist = euclidean_distances(self.uavs_pos, self.uavs_pos)
        dist[np.where(dist > self.r_buf)] = self.r_buf
        dist = np.sum(dist, axis=1)/(self.n_uavs - 1)
        
        return dist
    
        
    def cm_correction(self, matrix, const):
        scale = lambda x: x / (np.abs(x) / const)
        scale = np.vectorize(scale)
        
        return scale(matrix)
    
    
    def power_correction(self):
        self.hybrid_bf = np.einsum('abc,acd->abd', 
                                   self.analog_bf, self.digital_bf)
        
        F_norm = np.linalg.norm(self.hybrid_bf, axis=(1, 2))
        
        self.digital_bf /= (F_norm[:, None, None] / np.sqrt(self.power))
        
        self.hybrid_bf = np.einsum('abc,acd->abd', 
                                   self.analog_bf, self.digital_bf)
        
        return self.digital_bf
    
    
    def get_steer_vec(self):
        nx = np.repeat(np.arange(self.Nx), self.Ny).reshape((1, -1))
        ny = np.tile(np.arange(self.Ny), self.Nx).reshape((1, -1))
        
        azi = self.azi.reshape((-1, 1))
        ele = self.ele.reshape((-1, 1))
        
        # steer_vec = np.sin(azi) * (nx * np.cos(ele) + ny * np.sin(ele))
        steer_vec = np.sin(azi) * (np.dot(np.cos(ele), nx) + np.dot(np.sin(ele), ny))
        steer_vec = np.exp(1j * np.pi * steer_vec) / (Nx * Ny)
        steer_vec = steer_vec.reshape((self.n_uavs, self.n_users, -1))
        # steer_vec = steer_vec.swapaxes(1, 2)
        
        return steer_vec


    def get_ele(self):        
        X = self.uavs_pos[:, 0]
        Y = self.uavs_pos[:, 1]
        H = self.uavs_pos[:, 2]
        X = np.repeat(X, self.n_users).reshape((self.n_uavs, -1))
        Y = np.repeat(Y, self.n_users).reshape((self.n_uavs, -1))
        H = np.repeat(H, self.n_users).reshape((self.n_uavs, -1))
        
        x = self.users_pos[:, 0]
        y = self.users_pos[:, 1]
        
        return np.arctan(np.sqrt((X - x)**2 + (Y - y)**2) / H)


    def get_azi(self):
        X = self.uavs_pos[:, 0]
        Y = self.uavs_pos[:, 1]
        X = np.repeat(X, self.n_users).reshape((self.n_uavs, -1))
        Y = np.repeat(Y, self.n_users).reshape((self.n_uavs, -1))
        
        x = self.users_pos[:, 0]
        y = self.users_pos[:, 1]
        
        return np.arctan((Y - y)/(X - x))
    
    
    def collect_all_obs(self):
        joint_obs = []
        for m in range(self.n_uavs):
            m_obs = []
            m_obs.append(self.uavs_pos[m])
            m_obs.append(self.d0[m])
            m_obs.append(self.d_nearby[m])
            m_obs.append(self.served_users[m])
            m_obs.append(self.analog_bf[m])
            m_obs.append(self.digital_bf[m])
            
            joint_obs.append(m_obs)
            
        return joint_obs
    
    
    def get_d0(self):        
        return np.linalg.norm(self.uavs_pos[:, [0, 1]], axis=1)
    
    
    # def get_association(self):
        
    def get_association_cost(self):
        cost_p_LoS = 1 / self.p_LoS**2
        cost_dist = self.uavs_users_dist
        cost_fair = self.users_sumrate / self.users_sumrate.max()
        
        self.association_cost = cost_p_LoS * cost_dist * cost_fair
        
        return self.association_cost
    
    
    def get_association(self):
        self.association_cost = self.get_association_cost()
        # self.association_matrix = np.zeros((self.n_uavs, self.n_users))
        self.association = np.zeros((self.n_uavs, self.Nrf), dtype=np.uint8)
        x = {}
        for m in range(self.n_uavs):
            for k in range(self.n_users):
                x[m, k] = self.solver.IntVar(0, 1, '')

        for m in range(self.n_uavs):
            self.solver.Add(self.solver.Sum(
                [x[m, k] for k in range(self.n_users)]) == Nrf)

        for k in range(self.n_users):
            self.solver.Add(self.solver.Sum(
                [x[m, k] for m in range(self.n_uavs)]) <= 1)
            
        objective_terms = []
        for m in range(self.n_uavs):
            for k in range(self.n_users):
                objective_terms.append(self.association_cost[m][k] * x[m, k])
        self.solver.Minimize(self.solver.Sum(objective_terms))

        status = self.solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for m in range(n_uavs):
                i = 0
                for k in range(n_users):
                    if x[m, k].solution_value() > 0.5:
                        self.association[m][i] = k
                        i += 1
        
        else:            
            self.association = np.array(random.sample(
                range(self.n_users), self.n_uavs * self.Nrf)).reshape((self.n_uavs, -1))
        
        return self.association
        
    
    def get_FSPL(self):
        # uavs_pos = np.repeat(uavs_pos, self.Nrf).reshape((self.n_uavs, self.Nrf, -1))
        # uavs_users_dist = np.linalg.norm(uavs_pos - served_users, axis=2)
        
        self.FSPL = 20 * np.log10(self.uavs_users_dist * frequency) - 147.55
        
        return self.FSPL
    
    
    def get_p_LoS(self):
        # uavs_H = np.repeat(uavs_pos[:, 2], self.Nrf).reshape((self.n_uavs, -1))
        # uavs_pos = np.repeat(uavs_pos, self.Nrf).reshape((self.n_uavs, self.Nrf, -1))
        # uavs_users_dist = np.linalg.norm(uavs_pos - served_users, axis=2)
        uavs_H = self.uavs_pos[:, 2].repeat(self.n_users).reshape((self.n_uavs, -1))
        
        degree = np.arctan(uavs_H / self.uavs_users_dist) * 180 / np.pi
        
        self.p_LoS = 1 / (1 + C_1 * np.exp(-C_2 * (degree - C_1)))
                
        return self.p_LoS
    
    
    def get_PL(self):
        uavs_pos = np.tile(self.uavs_pos, self.n_users)
        uavs_pos = uavs_pos.reshape((self.n_uavs, self.n_users, -1))
        self.uavs_users_dist = np.linalg.norm(uavs_pos - self.users_pos, axis=2)
        
        self.p_LoS = self.get_p_LoS()
        self.FSPL = self.get_FSPL()
        
        LoS = self.p_LoS * (self.FSPL + zeta_LoS)
        NLoS = (1 - self.p_LoS) * (self.FSPL + zeta_NLoS)
        
        self.PL = LoS + NLoS
        
        return self.PL
    
    
    def get_h(self):
        self.h = np.zeros((self.n_uavs, self.n_users, self.n_antens), dtype=complex)
        self.PL = self.get_PL()
        for m in range(self.n_uavs):
            for k in range(self.n_users):
                self.h[m][k] = self.steer_vec[m][k] / np.sqrt(self.PL[m][k])
                
        return self.h
    
    def get_sinr(self):
        # sinr = np.zeros((self.n_uavs, self.Nrf))        
        # for m in range(self.n_uavs):
        #     for i in range(self.Nrf):
        #         h_mi = np.matrix(h[m][i].reshape((1, -1))).T.H
        #         sinr_mi = h_mi.dot(analog_bf[m]).dot(digital_bf[m].T[i])
        #         # ii = (m * self.Nrf) + i
        #         sinr[m][i] = sinr_mi.abs()**2
        
        # intra_noise = deepcopy(sinr)
        # intra_noise = intra_noise.sum(axis=1)[:, None] - intra_noise
        
        self.intended_signal = np.zeros((self.n_uavs, self.Nrf))
        for m in range(self.n_uavs):
            for i in range(self.Nrf):
                k = self.association[m][i]
                h_mi = self.h[m][k].conjugate()[None, :]
                sinr_mi = h_mi.dot(self.analog_bf[m]).dot(self.digital_bf[m][i])
                self.intended_signal[m][i] = np.abs(sinr_mi)**2
        
        self.intra_noise = self.intended_signal.sum(1)[:, None] - self.intended_signal   
        
        self.inter_signal = np.zeros((self.n_uavs * self.Nrf, self.n_uavs))
        for i, k in enumerate(self.association.flatten()):
            for m in range(n_uavs):
                h_mk = self.h[m][k].conjugate()[None, :]
                inter_sigmal_mk = h_mk.dot(self.analog_bf[m]).dot(self.digital_bf[m])
                self.inter_signal[i][m] = np.linalg.norm(inter_sigmal_mk)**2
                
        self.inter_signal = self.inter_signal.reshape((self.n_uavs, self.Nrf, -1))
        self.inter_noise = np.zeros((self.n_uavs, self.Nrf))
        for m in range(self.n_uavs):
            self.inter_noise[m] = self.inter_signal[m].sum(1) - self.inter_signal[m][:,m]
        
        self.sinr = self.intended_signal / (self.intra_noise + self.inter_noise + self.noise)
        
        return self.sinr
    
    
    def get_rdpe_reward(self, scale, rd_steep, re_steep):
        
        
    
    def reset(self):
        self.users_sumrate = np.zeros(self.n_users) + 1e-5
        
        self.uavs_battery = np.array([self.Emax] * self.n_uavs)
        
        uavs_random_x = np.random.uniform(0, 50, self.n_uavs)
        uavs_random_y = np.random.uniform(0, 50, self.n_uavs)
        uavs_random_H = np.random.uniform(self.Hmin, self.Hmax, self.n_uavs)
            
        users_random_x = np.random.uniform(0, self.Xmax, self.n_users)        
        users_random_y = np.random.uniform(0, self.Ymax, self.n_users)  
        
        self.uavs_pos = np.vstack((uavs_random_x, 
                                   uavs_random_y, 
                                   uavs_random_H)).T
        
        self.users_pos = np.vstack((users_random_x, 
                                    users_random_y, 
                                    np.zeros(self.n_users))).T
            
        self.d0 = self.get_d0()
        
        self.d_nearby = self.get_nearby_uav_dist()
                        
        
        self.azi = self.get_azi()
        self.ele = self.get_ele()
        self.steer_vec = self.get_steer_vec()
        
        self.h = self.get_h()
        
        self.association = self.get_association()  
        
        self.served_users = self.users_pos[self.association] 
        
        self.analog_bf = np.zeros((self.n_uavs, self.n_antens, self.Nrf), dtype=complex)
        for m in range(self.n_uavs):
            self.analog_bf[m] = self.steer_vec[m][self.association[m]].T
            
        self.analog_bf = self.cm_correction(self.analog_bf, 1/np.sqrt(self.n_antens))
        
        self.digital_bf = np.sqrt(self.power) * np.eye(self.Nrf)
        self.digital_bf = np.tile(self.digital_bf, 
                                  (self.n_uavs, 1)).reshape((-1, self.Nrf, self.Nrf))
        for m in range(self.n_uavs):
            self.digital_bf[m] /= np.linalg.norm(self.analog_bf[m])
        
        # self.digital_bf = self.digital_bf.astype(complex)
        self.digital_bf = self.power_correction()
        
        self.joint_obs = self.collect_all_obs()
                
        self.sinr = self.get_sinr()
        
        return self.joint_obs
        
    
    def step(self, spd, azi, ele, analog_bf, digital_bf):
        
        
        return
        
        