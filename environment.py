
import numpy as np
import configs as cf
from sklearn.metrics.pairwise import euclidean_distances


def const_modulus(cmplx_num, const_value):
    modulus = np.abs(cmplx_num)
    n = modulus/const_value
    
    return cmplx_num/n

def cm_correction(matrix, const):
    scaling = lambda x: x / ((np.abs(x) + 1e-6) / const)
    scaling = np.vectorize(scaling)
    
    return scaling(matrix)


class Environment():
    def __init__(self):
        
        self.Xmin, self.Xmax = cf.Xrange
        self.Ymin, self.Ymax = cf.Yrange
        self.diag_length = np.linalg.norm(np.array([self.Xmin, self.Ymin]) - np.array([self.Xmax, self.Ymax]))
        
        self.n_uavs = cf.n_uavs    
        self.n_users = cf.n_users
        
        self.n_antens = cf.n_antens
        self.Nx = cf.Nx
        self.Ny = cf.Ny
        self.Nrf = cf.Nrf
        self.power = cf.power
        
        self.Hmin, self.Hmax = cf.Hrange
        self.Vmax = cf.Vmax
        self.Emax = cf.Emax
        self.r_buf = cf.r_buf
        
        self.dt = cf.dt
        
        self.noise = cf.noise
        self.lambda_ = cf.c/cf.frequency
        
        self.destination = np.array([self.Xmax * 0.9, self.Ymax * 0.9])
        
        self.obs_dim = 3 + 1 + 1 + 1 + self.n_users * 2
            
        self.action_dim = 3 + self.n_users
        
        self.sumrate_weight = cf.sumrate_weight
        self.dest_reward_weight = cf.dest_reward_weight
        self.step_penalty_weight = cf.step_penalty_weight
        self.rd_steep = cf.rd_steep
        self.re_steep = cf.re_steep
        self.collision_penalty_weight = cf.collision_penalty_weight
        self.collision_penalty_steep = cf.collision_penalty_steep
        self.OOB_penalty_weight = cf.OOB_penalty_weight
        self.OOB_penalty_steep = cf.OOB_penalty_steep
        
        self.reset()
    
    
    def update_uavs_pos(self, V, azi, ele):
        dx = V * np.cos(azi) * np.cos(ele) * self.dt
        dy = V * np.sin(azi) * np.cos(ele) * self.dt
        dz = V * np.sin(ele) * self.dt
        
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        self.uavs_pos += np.vstack((dx, dy, dz)).T
        
        self.uavs_pos[:, 2][np.where(self.uavs_pos[:, 2] < self.Hmin)] = self.Hmin
        self.uavs_pos[:, 2][np.where(self.uavs_pos[:, 2] > self.Hmax)] = self.Hmax
        
        self.uavs_trajectory = np.hstack((self.uavs_trajectory, self.uavs_pos))
        
        return self.uavs_pos
    
    
    def cal_power_consumption(self, V):
        cost = cf.P_b * (1 + 3 * V**2/cf.U_tip**2) + \
               cf.P_i * np.sqrt( np.sqrt(1 + V**4/(4*cf.v_0**4)) - V**2/(2*cf.v_0**2) ) + \
               cf.d_0 * cf.rho * cf.s * cf.A * V**3/2
               
        return cost
    
    def update_uavs_battery(self, V):        
        cost = self.cal_power_consumption(V)
        
        self.uavs_battery -= cost*self.dt
        
        self.uavs_battery[np.where(self.uavs_battery < 0)] = 0.0
        
        return self.uavs_battery
    
    
    def uavs_move(self, V, azi, ele):
        self.uavs_pos = self.update_uavs_pos(V, azi, ele)
        self.uavs_battery = self.update_uavs_battery(V)
        
        return self.uavs_pos, self.uavs_battery
    
    
    def get_nearest_uav_dist(self):
        dist = euclidean_distances(self.uavs_pos, self.uavs_pos)
        dist[np.where(dist <= 0)] = np.inf
        dist = dist.min(axis=1)
        dist[np.where(dist > self.r_buf)] = self.r_buf
        
        return dist
    
    
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
        steer_vec = np.sin(azi) * (np.dot(np.cos(ele), nx) + np.dot(np.sin(ele), ny))
        steer_vec = np.exp(1j * np.pi * steer_vec) / (self.Nx * self.Ny)
        steer_vec = steer_vec.reshape((self.n_uavs, self.n_users, -1))
        
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
    
    
    def get_bf(self):
        self.analog_bf = np.zeros((self.n_uavs, self.n_antens, self.Nrf), dtype=complex)
        for m in range(self.n_uavs):
            self.analog_bf[m] = self.steer_vec[m][self.association[m]].T
            
        self.analog_bf = cm_correction(self.analog_bf, 1/np.sqrt(self.n_antens))
        
        self.digital_bf = np.sqrt(self.power) * np.eye(self.Nrf)
        self.digital_bf = np.tile(self.digital_bf, 
                                  (self.n_uavs, 1)).reshape((-1, self.Nrf, self.Nrf))
        for m in range(self.n_uavs):
            self.digital_bf[m] /= np.linalg.norm(self.analog_bf[m])
        
        self.digital_bf = self.power_correction()
        
        return self.analog_bf, self.digital_bf
    
    
    def collect_all_obs(self):        
        all_obs = []        
        for m in range(self.n_uavs):
            uav_pos = self.uavs_pos[m].flatten()
            d_nearest_uav = self.d_nearest_uav[m]
            d_destination = self.d_destination[m]
            battery = self.uavs_battery[m]
            users_pos = self.users_pos[:, :2].flatten()
            
            obs_m = np.hstack((uav_pos, d_nearest_uav, d_destination, battery, users_pos))
            all_obs.append(obs_m)
            
        all_obs = np.array(all_obs)
            
        # all_obs_dict = []
        # for m in range(self.n_uavs):
        #     obs_m_dict = {}
        #     obs_m_dict['pos'] = self.uavs_pos[m]
        #     obs_m_dict['d_nearest_uav'] = self.d_nearest_uav[m]
        #     obs_m_dict['d_destination'] = self.d_destination[m]
        #     obs_m_dict['battery'] = self.uavs_battery[m]
        #     obs_m_dict['users_pos'] = self.users_pos
            
        #     all_obs_dict.append(obs_m_dict)
        
        # self.all_obs = all_obs
        # self.all_obs_dict = all_obs_dict
            
        # return self.all_obs, self.all_obs_dict
        return all_obs
    
        
    def get_d_destination(self):        
        return np.linalg.norm(self.uavs_pos[:,:2] - self.destination, axis=1)
        
    
    def get_FSPL(self):        
        self.FSPL = 20 * np.log10(self.uavs_users_dist * cf.frequency) - 147.55
        
        return self.FSPL
    
    
    def get_p_LoS(self):
        uavs_H = self.uavs_pos[:, 2].repeat(self.n_users).reshape((self.n_uavs, -1))
        
        degree = np.arctan(uavs_H / self.uavs_users_dist) * 180 / np.pi
        
        self.p_LoS = 1 / (1 + cf.C_1 * np.exp(-cf.C_2 * (degree - cf.C_1)))
                
        return self.p_LoS
    
    
    def get_PL(self):
        uavs_pos = np.tile(self.uavs_pos, self.n_users)
        uavs_pos = uavs_pos.reshape((self.n_uavs, self.n_users, -1))
        self.uavs_users_dist = np.linalg.norm(uavs_pos - self.users_pos, axis=2)
        
        self.p_LoS = self.get_p_LoS()
        self.FSPL = self.get_FSPL()
        
        LoS = self.p_LoS * (self.FSPL + cf.zeta_LoS)
        NLoS = (1 - self.p_LoS) * (self.FSPL + cf.zeta_NLoS)
        
        self.PL = LoS + NLoS
        
        return self.PL
    
    
    def get_h(self):
        self.h = np.zeros((self.n_uavs, self.n_users, self.n_antens), dtype=complex)
        self.PL = self.get_PL()
        for m in range(self.n_uavs):
            for k in range(self.n_users):
                self.h[m][k] = self.steer_vec[m][k] / np.sqrt(self.PL[m][k])
                
        return self.h
    
    
    def get_SINR(self):        
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
            for m in range(self.n_uavs):
                h_mk = self.h[m][k].conjugate()[None, :]
                inter_sigmal_mk = h_mk.dot(self.analog_bf[m]).dot(self.digital_bf[m])
                self.inter_signal[i][m] = np.linalg.norm(inter_sigmal_mk)**2
                
        self.inter_signal = self.inter_signal.reshape((self.n_uavs, self.Nrf, -1))
        self.inter_noise = np.zeros((self.n_uavs, self.Nrf))
        for m in range(self.n_uavs):
            self.inter_noise[m] = self.inter_signal[m].sum(1) - self.inter_signal[m][:,m]
        
        self.sinr = self.intended_signal / (self.intra_noise + self.inter_noise + self.noise)
        
        return self.sinr
    
    
    def get_rate(self):
        self.rate = np.log2(1 + self.sinr)
        
        return self.rate
        
    
    def get_step_penalty(self, rd_steep, re_steep):
        remaining_energy = self.uavs_battery / self.Emax
        remaining_distance = self.d_destination / self.diag_length
        result = 1 + 1 / np.exp(re_steep * remaining_energy) - 1 / np.exp(rd_steep * remaining_distance)
        
        return result
    
    def get_collision_penalty(self, steep):
        self.d_nearest_uav = self.get_nearest_uav_dist()        
        collision_pen = 1 / np.exp(steep * self.d_nearest_uav/self.r_buf)
        
        return collision_pen
    
    
    def get_OOB_penalty(self):
        OOB_X = np.zeros(self.n_uavs)        
        OOB_Y = np.zeros(self.n_uavs)
        OOB_Z = np.zeros(self.n_uavs)
        
        uavs_X = self.uavs_pos[:, 0]
        uavs_Y = self.uavs_pos[:, 1]
        uavs_Z = self.uavs_pos[:, 2]
        
        OOB_X[uavs_X < 0] = -uavs_X[uavs_X < 0]
        OOB_Y[uavs_Y < 0] = -uavs_Y[uavs_Y < 0]
        OOB_X[uavs_X > self.Xmax] = uavs_X[uavs_X > self.Xmax] - self.Xmax        
        OOB_X[uavs_Y > self.Ymax] = uavs_Y[uavs_Y > self.Ymax] - self.Ymax
        
        OOB_Z[uavs_Z < self.Hmin] = self.Hmin - uavs_Z[uavs_Z < self.Hmin]
        OOB_Z[uavs_Z > self.Hmax] = uavs_Z[uavs_Z > self.Hmax] - self.Hmax
        
        Z_range = self.Hmax - self.Hmin
        
        oob_dist = np.sqrt((OOB_X**2 + OOB_Y**2 + OOB_Z**2) / (self.Xmax**2 + self.Ymax**2 + Z_range**2))
        
        return oob_dist
    
    
    def reset(self):        
        self.uavs_battery = np.array([self.Emax] * self.n_uavs, dtype=np.float64)
        
        uavs_random_x = np.random.uniform(0, 50, self.n_uavs)
        uavs_random_y = np.random.uniform(0, 50, self.n_uavs)
        uavs_random_H = np.random.uniform(self.Hmin, self.Hmax, self.n_uavs)
        
        # uavs_x = np.array([0.0, self.Xmax, 0.0], dtype=np.float64)
        # uavs_y = np.array([0.0, 0.0, self.Ymax], dtype=np.float64)
        # uavs_H = np.array([(self.Hmin + self.Hmax)/2] * 3, dtype=np.float64)
        
        # uavs_x = np.array([0.0, 0.0], dtype=np.float64)
        # uavs_y = np.array([0.0, 0.0], dtype=np.float64)
        # uavs_H = np.array([(self.Hmin + self.Hmax)/2] * 2, dtype=np.float64)
            
        users_random_x = np.random.uniform(0, self.Xmax, self.n_users)        
        users_random_y = np.random.uniform(0, self.Ymax, self.n_users)
        
        self.uavs_pos = np.vstack((uavs_random_x, 
                                   uavs_random_y, 
                                   uavs_random_H)).T
        
        self.users_pos = np.vstack((users_random_x, 
                                    users_random_y, 
                                    np.zeros(self.n_users))).T
        
        self.users_pos = self.users_pos.astype(np.float64)
        
        self.association = np.random.choice(self.n_users, (self.n_uavs, self.Nrf))
        
        self.uavs_trajectory = self.uavs_pos
            
        self.d_destination = self.get_d_destination()
        
        self.d_nearest_uav = self.get_nearest_uav_dist()                        
        
        self.current_step = 0
        
        # self.all_obs, self.all_obs_dict = self.collect_all_obs()
        self.all_obs = self.collect_all_obs()
        
        return self.all_obs
        
    
    def step(self, spd, azi, ele, association):
        self.current_step += 1
        
        self.association = association
        
        self.uavs_pos, self.uavs_battery = self.uavs_move(spd, azi, ele)
        
        old_d_destination = self.d_destination        
        self.d_destination = self.get_d_destination()
        self.dest_reward = self.dest_reward_weight * (old_d_destination - self.d_destination)
        
        self.d_nearest_uav = self.get_nearest_uav_dist()
        
        self.azi = self.get_azi()
        self.ele = self.get_ele()
        self.steer_vec = self.get_steer_vec()
        
        self.h = self.get_h()
            
        self.analog_bf, self.digital_bf = self.get_bf()
        self.analog_bf = cm_correction(self.analog_bf, 1/np.sqrt(self.n_antens))
        self.digital_bf = self.power_correction()
        
        self.sinr = self.get_SINR()
        self.rate = self.get_rate()
        
        self.sumrate = self.rate.sum(axis=1)
        self.step_penalty = self.step_penalty_weight * self.get_step_penalty(self.rd_steep, self.re_steep) # 3
        self.collision_penalty = self.collision_penalty_weight * self.get_collision_penalty(self.collision_penalty_steep) # 200, 5
        self.OOB_penalty = self.OOB_penalty_weight * self.get_OOB_penalty() # 15
        
        self.step_reward = \
                self.sumrate_weight * self.sumrate + \
                self.dest_reward - \
                self.step_penalty - \
                self.collision_penalty - \
                self.OOB_penalty
                
        penalties = [self.dest_reward, self.step_penalty, self.collision_penalty, self.OOB_penalty]
        
        # self.all_obs, self.all_obs_dict = self.collect_all_obs()
        self.all_obs = self.collect_all_obs()
        
        return self.sumrate, self.step_reward, self.all_obs, penalties
        
        