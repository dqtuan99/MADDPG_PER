# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:49:47 2022

@author: TUAN
"""


import numpy as np

class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-3, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)