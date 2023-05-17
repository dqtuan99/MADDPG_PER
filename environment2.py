
import numpy as np
import configs as cf
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


def const_modulus(cmplx_num, const_value):
    modulus = np.abs(cmplx_num)
    n = modulus/const_value
    
    return cmplx_num/n

def cm_correction(matrix, const):
    scale = lambda x: x / ((np.abs(x) + 1e-6) / const)
    scale = np.vectorize(scale)
    
    return scale(matrix)


class Environment():
    def __init__(self):
        
        self.Xmin, self.Xmax = cf.Xrange
        self.Ymin, self.Ymax = cf.Yrange
        
        self.n_uavs = cf.n_uavs
        
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
    
        self.n_users = cf.n_users
        
        self.noise = cf.noise
        self.lambda_ = cf.c/cf.frequency
        
        self.ending_point = np.array([self.Xmax, self.Ymax])
        
        self.obs_dim = 3 + 1 + 1 + 1 + 2 * self.Nrf
            
        self.action_dim = 1 + 1 + 1
        
        self.reset()
        
    
    def check_uavs_done(self):        
        for m in range(self.n_uavs):
            if self.uavs_battery[m] < 0:
                self.uavs_done[m] = True
            elif self.d0[m] < 20:
                self.uavs_done[m] = True
        
        return self.uavs_done
    
    
    def update_uavs_pos(self, V, azi, ele):
        self.uavs_done = self.check_uavs_done()
        dx = V * np.cos(azi) * np.cos(ele) * self.dt * (1 - self.uavs_done)
        dy = V * np.sin(azi) * np.cos(ele) * self.dt * (1 - self.uavs_done)
        dz = V * np.sin(ele) * self.dt * (1 - self.uavs_done)
        
        self.uavs_pos += np.vstack((dx, dy, dz)).T
        
        self.uavs_pos[:, 0][np.where(self.uavs_pos[:, 0] < self.Xmin)] = self.Xmin
        self.uavs_pos[:, 0][np.where(self.uavs_pos[:, 0] > self.Xmax)] = self.Xmax
        
        self.uavs_pos[:, 1][np.where(self.uavs_pos[:, 1] < self.Ymin)] = self.Ymin
        self.uavs_pos[:, 1][np.where(self.uavs_pos[:, 1] > self.Ymax)] = self.Ymax
        
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
        steer_vec = np.exp(1j * np.pi * steer_vec) / (self.Nx * self.Ny)
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
            d0 = self.d0[m]
            d_nearby_uavs = self.d_nearby_uavs[m]
            battery = self.uavs_battery[m]
            served_users_pos = self.served_users_pos[m].flatten()
            
            obs_m = np.hstack((uav_pos, d0, d_nearby_uavs, battery, served_users_pos))
            all_obs.append(obs_m)
            
        all_obs = np.array(all_obs)
            
        all_obs_dict = []
        for m in range(self.n_uavs):
            obs_m_dict = {}
            obs_m_dict['pos'] = self.uavs_pos[m]
            obs_m_dict['d0'] = self.d0[m]
            obs_m_dict['d_nearby_uavs'] = self.d_nearby_uavs[m]
            obs_m_dict['battery'] = self.uavs_battery[m]
            obs_m_dict['served_users_pos'] = self.served_users_pos[m]
            
            all_obs_dict.append(obs_m_dict)
        
        self.all_obs = all_obs
        self.all_obs_dict = all_obs_dict
            
        return self.all_obs, self.all_obs_dict
    
        
    def get_d0(self):        
        return np.linalg.norm(self.uavs_pos[:,:2] - self.ending_point, axis=1)
    
    
    def get_user_clusters(self):
        users_2D_pos = self.users_pos[:, :2]
        kmeans = KMeans(n_clusters=self.n_uavs, random_state=0).fit(users_2D_pos)
        self.cluster_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        
        return self.cluster_labels
    
    
    def get_association(self):
        self.association = np.zeros((self.n_uavs, self.Nrf), dtype=int)
        for m in range(self.n_uavs):
            cluster_users_idx = np.where(self.cluster_labels == m)[0]
            cluster_users_pos = self.users_pos[cluster_users_idx]
            dist = np.zeros_like(cluster_users_idx, dtype=np.float32)
            
            for i, user_idx in enumerate(cluster_users_idx):
                dist[i] = np.linalg.norm(self.uavs_pos[m] - cluster_users_pos[i])
            
            nearest_users_i = np.argpartition(dist, self.Nrf)[:self.Nrf]
            nearest_users_idx = cluster_users_idx[nearest_users_i]
            
            self.association[m] = nearest_users_idx
        
        return self.association
        
    
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
    
    
    def get_RDPE_reward(self, scale, rd_steep, re_steep):
        rd = self.uavs_pos[:, :2]**2
        rd = np.sum(rd, axis=1)
        rd = np.sqrt(rd / (self.Xmax**2 + self.Ymax**2))
        
        re = self.uavs_battery / self.Emax
        
        r_rdpe = scale / (np.exp(rd_steep * rd) * np.exp(re_steep * re))
        
        return r_rdpe, rd, re
    
    
    def get_collision_penalty(self, scale, steep):
        self.d_nearby_uavs = self.get_nearby_uav_dist()        
        collision_pen = scale / np.exp(steep * self.d_nearby_uavs/self.r_buf)
        
        return collision_pen
    
    
    def reset(self):
        self.users_sumrate = np.zeros(self.n_users) + 1e-5
        
        self.uavs_battery = np.array([self.Emax] * self.n_uavs)
        self.uavs_done = self.uavs_battery < 0
        
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
        
        self.uavs_trajectory = self.uavs_pos
            
        self.d0 = self.get_d0()
        
        self.d_nearby_uavs = self.get_nearby_uav_dist()
                        
        
        self.azi = self.get_azi()
        self.ele = self.get_ele()
        self.steer_vec = self.get_steer_vec()
        
        self.h = self.get_h()
        
        self.cluster_labels = self.get_user_clusters()
        
        self.association = self.get_association()  
        
        self.served_users_pos = self.users_pos[self.association][:,:,:2]
        
        self.analog_bf, self.digital_bf = self.get_bf()
        
        self.all_obs, self.all_obs_dict = self.collect_all_obs()
                
        self.sinr = self.get_SINR()
        
        self.current_step = 0
        
        # self.reached_cluster_centers = [False] * self.n_uavs
        
        return self.all_obs
    
    
    # def get_dist_to_clusters(self):
    #     self.dist_to_clusters = np.zeros(self.n_uavs)
        
    #     for m in range(self.n_uavs):
    #         self.dist_to_clusters[m] = np.linalg.norm(self.uavs_pos[m, :2] - self.cluster_centers[m])
    #         if self.dist_to_clusters[m] < self.Vmax * self.dt:
    #             self.reached_cluster_centers[m] = True
        
    #     return self.dist_to_clusters
    
    
    # def move_to_cluster_centers(self):
    #     spd = np.zeros(self.n_uavs, dtype=np.float32)
    #     azi = np.zeros(self.n_uavs, dtype=np.float32)
    #     ele = np.zeros(self.n_uavs, dtype=np.float32)
        
    #     self.dist_to_clusters = self.get_dist_to_clusters()
        
    #     for m in range(self.n_uavs):
    #         if (self.reached_cluster_centers[m]):
    #             continue
            
    #         spd[m] = min(self.dist_to_clusters[m] / self.dt, self.Vmax)
            
    #         X = self.uavs_pos[m, 0]
    #         Y = self.uavs_pos[m, 1]            
    #         x = self.cluster_centers[m, 0]
    #         y = self.cluster_centers[m, 1]            
    #         azi[m] = np.arctan((Y - y)/(X - x))
        
    #     return spd, azi, ele
    
    
    def required_step_to_reach_endpoint(self, V):        
        reach_endpoint_time = self.d0 / V
        reach_endpoint_step = np.ceil(reach_endpoint_time / self.dt)\
            
        return reach_endpoint_step
        
    
    def required_energy_to_reach_endpoint(self, V):
        power_consumption = self.cal_power_consumption(V)
        reach_endpoint_step = self.required_step_to_reach_endpoint(V)
        
        return power_consumption * reach_endpoint_step
    
    
    def trigger_rush_mode(self):
        self.critical_level = self.required_energy_to_reach_endpoint(18)
        self.rush_mode = self.uavs_battery + 1000 < self.critical_level
        
        return self.rush_mode
    
    
    def get_rush_mode_azi(self):
        azi = np.zeros(self.n_uavs, dtype=np.float32)
        for m in range(self.n_uavs):
            X = self.uavs_pos[m, 0]
            Y = self.uavs_pos[m, 1]
            x = self.ending_point[0]
            y = self.ending_point[1]
            azi[m] = np.arctan((Y - y) / (X - x))
        
        return azi
        
    
    def step(self, spd, azi, ele):
        self.current_step += 1
        
        self.uavs_pos, self.uavs_battery = self.uavs_move(spd, azi, ele)
        
        self.d0 = self.get_d0()
        
        self.d_nearby_uavs = self.get_nearby_uav_dist()
        
        self.azi = self.get_azi()
        self.ele = self.get_ele()
        self.steer_vec = self.get_steer_vec()
        
        self.h = self.get_h()
        
        self.association = self.get_association()        
        self.served_users_pos = self.users_pos[self.association][:,:,:2]
            
        self.analog_bf, self.digital_bf = self.get_bf()
        self.analog_bf = cm_correction(self.analog_bf, 1/np.sqrt(self.n_antens))
        self.digital_bf = self.power_correction()
        
        self.sinr = self.get_SINR()
        self.rate = self.get_rate()
        
        self.sumrate = self.rate.sum()
        self.RDPE_reward, self.RD , self.RE = self.get_RDPE_reward(5, 1, 1)
        self.collision_penalty = self.get_collision_penalty(3, 1)
        
        self.step_reward = self.sumrate + self.RDPE_reward - self.collision_penalty
        self.other_rewards = (self.RDPE_reward, self.collision_penalty)        
        
        self.all_obs, self.all_obs_dict = self.collect_all_obs()
        
        return self.sumrate, self.step_reward, self.all_obs, self.uavs_done, self.other_rewards
        
        