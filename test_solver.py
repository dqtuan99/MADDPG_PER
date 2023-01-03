# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 06:07:33 2022

@author: TUAN
"""

import numpy as np
from ortools.linear_solver import pywraplp

n_uavs = 3
n_users = 8
Nrf = 2

costs = np.random.uniform(0, 10, (n_uavs, n_users))

solver = pywraplp.Solver.CreateSolver('SCIP')

x = {}
for i in range(n_uavs):
    for j in range(n_users):
        x[i, j] = solver.IntVar(0, 1, '')

for i in range(n_uavs):
    solver.Add(solver.Sum([x[i, j] for j in range(n_users)]) == Nrf)

for j in range(n_users):
    solver.Add(solver.Sum([x[i, j] for i in range(n_uavs)]) <= 1)
    
objective_terms = []
for i in range(n_uavs):
    for j in range(n_users):
        objective_terms.append(costs[i][j] * x[i, j])
solver.Minimize(solver.Sum(objective_terms))

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    print(f'Total costs = {solver.Objective().Value()}\n')
    for i in range(n_uavs):
        for j in range(n_users):
            # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
            if x[i, j].solution_value() > 0.5:
                print(f'Uav {i} assigned to user {j}.' +
                      f' Cost: {costs[i][j]}')
else:
    print('No solution found.')