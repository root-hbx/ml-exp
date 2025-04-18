import os
import time
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

import numpy as np
from pprint import pprint

from tsp import *
import tb
import ga
import sa


if not os.path.exists('results'):
    os.makedirs('results')

# load data
pos = [[float(x) for x in s.split()[1:]] for s in open('data/dj38.txt').readlines()]
n = len(pos)


# calculate adjacency matrix

adj_mat = np.zeros([n, n])
for i in range(n):
    for j in range(i, n):
        adj_mat[i][j] = adj_mat[j][i] = np.linalg.norm(np.subtract(pos[i], pos[j]))


# initialization

opt_cost = 6659.439330623091  # get result from tsp_gurobi.py
num_tests = 1  # number of iid tests
result = {'best_sol': [], 'best_cost': math.inf, 'best_gap': math.inf,
          'cost': [0] * num_tests, 'time': [0] * num_tests,
          'avg_cost': math.inf, 'avg_gap': math.inf, 'cost_std': math.inf,
          'avg_time': math.inf, 'time_std': math.inf}
best_cost = math.inf
best_sol = []
data = {}


# set method

method = 'ts'  # tabu search
# method = 'ga'  # genetic algorithm
# method = 'sa'  # simulated annealing


# set mutation method

# mut_md = [get_new_sol_swap, get_delta_swap]
mut_md = [get_new_sol_2opt, get_delta_2opt]


# run and visualization

method_name = ''
for _ in tqdm(range(num_tests)):
    start = time.time()
    if method == 'ts':
        method_name = 'Tabu Search'
        best_sol, best_cost, data = tb.tb(n, adj_mat,tb_size=20, max_tnm=100, 
                                    mut_md=mut_md, term_count=200)
    elif method == 'ga':
        method_name = 'Genetic Algorithm'
        best_sol, best_cost, data = ga.ga(n, adj_mat, n_pop=200, r_cross=0.5, 
                                    r_mut=0.8, selection_md='tnm', max_tnm=3, term_count=200)
    elif method == 'sa':
        method_name = 'Simulated Annealing'
        best_sol, best_cost, data = sa.sa(n, adj_mat, tb_size=0, max_tnm=20, mut_md=mut_md,
                                    term_count_1=25, term_count_2=25, t_0=1200, alpha=0.9)
    else:
        assert 0, 'unknown method'

    end = time.time()
    result['time'][_] = end - start
    result['cost'][_] = best_cost

    if best_cost < result['best_cost']:
        result['best_sol'] = best_sol
        result['best_cost'] = best_cost
        result['best_gap'] = best_cost / opt_cost - 1

    plt.plot(range(len(data['cost'])), data['cost'], color='b', alpha=math.pow(num_tests, -0.75))
    plt.plot(range(len(data['cost'])), data['best_cost'], color='r', alpha=math.pow(num_tests, -0.75))


plt.title('Solving TSP with {}'.format(method_name))
plt.xlabel('Number of Iteration')
plt.ylabel('Cost')
plt.savefig('results/{}.png'.format(method))

# print results
result['avg_cost'] = np.mean(result['cost'])
result['avg_gap'] = result['avg_cost'] / opt_cost - 1
result['cost_std'] = np.std(result['cost'])
result['avg_time'] = np.mean(result['time'])
result['time_std'] = np.std(result['time'])
pprint(result)


# SA visualization
if num_tests == 1:
    if method_name == 'Simulated Annealing':
        fig, ax = plt.subplots(figsize=(8, 6))

        final_sol = data['best_sol'][-1]

        xlim = [np.min(pos, 0)[0], np.max(pos, 0)[0]]
        ylim = [np.min(pos, 0)[1], np.max(pos, 0)[1]]
        ax.set(xlabel='X Axis', ylabel='Y Axis',
            xlim=xlim, ylim=ylim,
            title='Optimal Tour with Simulated Annealing')

        ax.scatter([p[0] for p in pos], [p[1] for p in pos], c='black', s=30)

        lines = [[pos[final_sol[i]], pos[final_sol[(i + 1) % n]]] for i in range(n)]
        line_segments = LineCollection(lines, color='r', linewidth=1.5)
        ax.add_collection(line_segments)

        plt.savefig('results/sa_path.png', dpi=300, bbox_inches='tight')
        plt.close()
    elif method_name == 'Tabu Search':
        fig, ax = plt.subplots(figsize=(8, 6))

        final_sol = data['best_sol'][-1]

        xlim = [np.min(pos, 0)[0], np.max(pos, 0)[0]]
        ylim = [np.min(pos, 0)[1], np.max(pos, 0)[1]]
        ax.set(xlabel='X Axis', ylabel='Y Axis',
            xlim=xlim, ylim=ylim,
            title='Optimal Tour with Simulated Annealing')

        ax.scatter([p[0] for p in pos], [p[1] for p in pos], c='black', s=30)

        lines = [[pos[final_sol[i]], pos[final_sol[(i + 1) % n]]] for i in range(n)]
        line_segments = LineCollection(lines, color='r', linewidth=1.5)
        ax.add_collection(line_segments)

        plt.savefig('results/ts_path.png', dpi=300, bbox_inches='tight')
        plt.close()
    elif method_name == 'Genetic Algorithm':
        fig, ax = plt.subplots(figsize=(8, 6))

        final_sol = data['best_sol'][-1]

        xlim = [np.min(pos, 0)[0], np.max(pos, 0)[0]]
        ylim = [np.min(pos, 0)[1], np.max(pos, 0)[1]]
        ax.set(xlabel='X Axis', ylabel='Y Axis',
            xlim=xlim, ylim=ylim,
            title='Optimal Tour with Simulated Annealing')

        ax.scatter([p[0] for p in pos], [p[1] for p in pos], c='black', s=30)

        lines = [[pos[final_sol[i]], pos[final_sol[(i + 1) % n]]] for i in range(n)]
        line_segments = LineCollection(lines, color='r', linewidth=1.5)
        ax.add_collection(line_segments)

        plt.savefig('results/ga_path.png', dpi=300, bbox_inches='tight')
        plt.close()
