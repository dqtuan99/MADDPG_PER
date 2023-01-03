# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 05:47:39 2022

@author: TUAN
"""

import numpy as np


def a(Nx, Ny, azi, ele):
    # xy_pairs = np.array(np.meshgrid(np.arange(Nx), np.arange(Ny))).T.reshape((-1, 2))
    nx = np.repeat(np.arange(Nx), Ny)
    ny = np.tile(np.arange(Ny), Nx)
    
    steer_vec = np.sin(azi) * (nx * np.cos(ele) + ny * np.sin(ele))
    steer_vec = np.exp(1j * np.pi * steer_vec) / (Nx * Ny)
    
    return steer_vec


def calc_ele(u, v):
    x, y, h = u
    X, Y, H = v
    
    return np.arctan(np.sqrt((X - x)**2 + (Y - y)**2) / (H - h))


def calc_azi(u, v):
    x, y, _ = u
    X, Y, _ = v
    
    return np.arctan((Y - y)/(X - x))