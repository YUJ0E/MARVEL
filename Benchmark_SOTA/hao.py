#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:34:28 2023

@author: dreamone
"""

import pandas as pd
import numpy as np
import os
import sys
import time as systime


class Network:
    def __init__(self, node_num=0, edge_num=0, name=''):
        self.name = name
        self.node_num = node_num
        self.edge_num = edge_num
        self.matrix = np.full((self.node_num, self.node_num), -1, dtype=int)  # Adjacency matrix
        self.index = np.full((self.edge_num, 2), -1, dtype=int)  # Start and end of the edge
        self.distrib = np.zeros((edge_num, 2))  # MU and SIGMA of the edge
        self.test_od = None  # OD pairs for test

    # Initial the network by the CSV data
    def get_init(self, nu=0.4):
        file_net_path = os.path.dirname(os.getcwd()) + f'/Networks/{self.name}/{self.name}_network.csv'
        net_data = pd.read_csv(file_net_path).values
        if len(net_data) != self.edge_num:
            print('Edge number is NOT equal to the lenth of Data !!!!!!!!!!!')
            sys.exit()
        else:
            # Numbered from 1, but stored from 0
            bias = 0 if min(net_data[:, 0]) == 0 or min(net_data[:, 1]) == 0 else 1
            for i in range(len(net_data)):
                self.matrix[int(net_data[i][0]) - bias, int(net_data[i][1]) - bias] = i
                self.index[i] = [int(net_data[i][0]) - bias, int(net_data[i][1]) - bias]
            self.distrib[:, 0] = net_data[:, 3]
            # Sigma is generated randomly based on the NU
            file_sigma = os.path.dirname(os.getcwd()) + f'/Networks/{self.name}/{self.name}_{nu}_random_sigma.npy'
            if os.path.exists(file_sigma):
                sigma = np.load(file_sigma)
                print('Sigma file exists!')
            else:
                print('Sigma file do not exist，and sigma will be generated!')
                sigma = self.distrib[:, 0] * (np.random.rand(len(self.distrib[:, 0])) * nu)
                np.save(file_sigma, sigma)
            self.distrib[:, 1] = sigma
        # Test OD pairs load by CSV data
        file_od_path = os.path.dirname(os.getcwd()) + f'/Networks/{self.name}/{self.name}_OD.csv'
        od_data = pd.read_csv(file_od_path).values
        self.test_od = od_data[:, 1:3].astype(int).tolist()


class LET:
    def __init__(self, network, origin, destination):
        self.cost_all, self.sigma_all, self.path = self._get_shortest_path(network, origin, destination)
        self.mu = network.distrib[self.path, 0].sum().item()
        self.sigma = np.sqrt((network.distrib[self.path, 1] ** 2).sum()).item()
        self.cost = self.cost_all[origin]
        self.path_print = [x + 1 for x in self.path]

    # Bellman-Ford algorithms for solving shortest paths
    def _get_shortest_path(self, network, origin, destination):
        # Reverse the direction of the edge
        revers_idx = np.zeros_like(network.index, dtype=int)
        revers_idx[:, 0] = network.index[:, 1]
        revers_idx[:, 1] = network.index[:, 0]
        # Solve shortest distance by Bellman-Ford algorithms
        distance = np.full((network.node_num,), np.inf)
        distance[destination] = 0
        variation = True
        while variation is True:
            add = network.distrib[:, 0] + distance[revers_idx[:, 0]]
            min_idx = add < distance[revers_idx[:, 1]]
            if variation in min_idx:
                distance[revers_idx[min_idx, 1]] = add[min_idx]
            else:
                variation = False
        # Solve shortest path by distance
        trace = [[] for _ in range(network.node_num)]
        dist = np.empty_like(distance)
        dist[:] = distance[:]
        min_trace = destination
        for i in range(len(distance) - 1):
            dist[min_trace] = float('inf')
            min_dist_idx = np.argmin(dist)
            neig_idx = network.index[:, 0] == min_dist_idx
            neig_dist = (distance[network.index[neig_idx, 1]] + network.distrib[neig_idx, 0]) == distance[min_dist_idx]
            path = np.arange(network.edge_num)[neig_idx][neig_dist]
            if len(path) != 1:
                path = path[0]
            trace[min_dist_idx] = [_ for _ in trace[network.index[path, 1].item()]]
            trace[min_dist_idx].insert(0, path.item())
            min_trace = min_dist_idx
        sigma = [np.sqrt((network.distrib[x, 1] ** 2).sum()) if x != [] else 0 for x in trace]
        return distance, sigma, trace[origin]


class PQL():
    def __init__(self, network):
        Net_list = ['SiouxFalls', 'Friedrichshain', 'Anaheim', 'Chicago_Sketch', 'Winnipeg', 'Barcelona']
        Net_node_edge_num_list = [[24, 76], [224, 531], [416, 914], [933, 2950], [1052, 2836], [1020, 2522]]
        net_idx = Net_list.index(network)
        net_node_edge_num = Net_node_edge_num_list[net_idx]
        Node, Edge = net_node_edge_num[0], net_node_edge_num[1]
        self.network = Network(Node, Edge, network)
        self.network.get_init(nu=0.4)
        
    def train(self,od, budget=1, alpha=1e-6, Epoch_test=5000):
        Origin, Destination = od[0] - 1, od[1] - 1
        let = LET(self.network, Origin, Destination)
        policy = np.zeros((self.network.edge_num, 2))
        let.sigma_all[Destination] = min(self.network.distrib[:, 1])
        policy[:, 1] = np.array(let.sigma_all)[self.network.index[:, 1]]
        policy[:, 0] = let.cost_all[self.network.index[:, 1]]
        Time = Time = let.cost * budget
        print('*' * 3, ' Test before Train', '*' * 3)
        accuracy, cost_time, path_list = self._get_test(self.network, policy, Origin, Destination, Time, Epoch_test)
        print(f'Test Times:{Epoch_test}   Accuracy:{(accuracy* 100):.2f}%   Cost Time:{cost_time}s')
        return accuracy, cost_time, path_list
    
    
    def _get_train(network, policy, origin, destination, Time, batch):
        # Prevent overflow
        def get_overflow_mask(data):
            limit = 700  # exp(double-float) can handle exponentials typically in the range of -709 to 709
            index = data > limit
            data[index] = limit
            index = data < -(limit)
            data[index] = -(limit)
            return data
        # Average over multiple samples
        delta = np.zeros_like(policy)
        for batch_idx in range(batch):
            observ = np.random.normal(network.distrib[:, 0], network.distrib[:, 1], size=len(network.distrib))
            time, path = 0, origin
            Prob, Hot = np.zeros(network.edge_num), np.zeros(network.edge_num)
            Observ, Remain_Time = np.zeros(network.edge_num), np.zeros(network.edge_num)
            Observ[:] = observ[:]
            # A trajectory to destination
            while time <= Time and path != destination:
                neig_idx = network.index[network.index[:, 0] == path, 1]
                path_idx = network.matrix[path, neig_idx]
                # Determine next steps by sampling
                theta1, theta2, ob = policy[path_idx, 0], policy[path_idx, 1], observ[path_idx]
                remain_time = Time - time
                exponent = get_overflow_mask(theta2 * (remain_time - ob) - theta1)
                p = np.exp(exponent) / np.exp(exponent).sum()
                next_idx = path_idx[np.where(np.random.rand() <= np.cumsum(p))[0][0]]

                time += observ[next_idx]
                path = np.where(next_idx == network.matrix[path, :])[0][0]
                observ[path_idx] = np.random.normal(network.distrib[path_idx, 0], network.distrib[path_idx, 1])
                Remain_Time[path_idx] = remain_time
                Prob[path_idx] = p
                Hot[path_idx] = 0
                Hot[next_idx] = 1
            if time <= Time and path == destination:
                r = 1
            else:
                r = 0
            delta[:, 0] += (r - 1 / 10) * ((-1) * (Hot - Prob))
            delta[:, 1] += (r - 1 / 10) * (Remain_Time - Observ) * (Hot - Prob)
        return delta / batch


    def _get_test(self, network, policy, origin, destination, Time, epoch):
        file_observ = os.path.dirname(os.getcwd()) + f'\\Networks\\{network.name}\\{network.name}_{epoch}_random_observation.csv'
        observ_epoch = np.loadtxt(file_observ, delimiter=",")
        count = 0
        start = systime.perf_counter()

        for i in range(epoch):
            observ = observ_epoch[i]
            path_list = [origin + 1]
            time, path = 0, origin

            while time <= Time and path != destination:
                neig_idx = network.index[network.index[:, 0] == path, 1]
                path_idx = network.matrix[path, neig_idx]
                theta1, theta2, ob = policy[path_idx, 0], policy[path_idx, 1], network.distrib[path_idx]
                remain_time = Time - time
                next_idx = path_idx[np.argmax((remain_time - ob[:,0] - theta1) / np.sqrt(theta2 ** 2 + ob[:,1] ** 2))]
                time += observ[next_idx]
                path = np.where(next_idx == network.matrix[path, :])[0][0].item()
                path_list.append(path + 1)

            if time <= Time and path == destination:
                count += 1

        cost_time = systime.perf_counter() - start
        print("path_list: ", path_list)
        return count / epoch, cost_time / epoch * 5, path_list

