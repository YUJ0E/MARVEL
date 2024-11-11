import networkx as nx
import numpy as np
import pandas as pd
import os


def gen_M(net):
    prefix = os.path.dirname(os.getcwd())
    file_path = prefix + f'\\Networks\\{net}\\{net}_network.csv'
    # file_path = prefix + f'/nn/{net}/{net}_network.csv'
    raw_data = pd.read_csv(file_path)
    origins = raw_data['From']
    destinations = raw_data['To']
    if origins.min() == 0 or destinations.min() == 0:
        origins += 1
        destinations += 1
    n_node = max(origins.max(), destinations.max())
    n_link = raw_data.shape[0]

    M = np.zeros((n_node, n_link))
    for i in range(n_link):
        M[origins[i] - 1, i] = 1
        M[destinations[i] - 1, i] = -1
    mu = np.array(raw_data['Cost']).reshape(-1, 1)
    nu = 0.4
    file_sigma = prefix + f'/Networks/{net}/{net}_{nu}_random_sigma.npy'
    if os.path.exists(file_sigma):
        sigma = np.load(file_sigma).reshape(-1, 1)
        print('Sigma文件已存在！')
    else:
        print('Sigma文件不存在，将随机创建Sigma')
        sigma = mu * (nu * np.random.rand(len(mu)).reshape(-1, 1))
        np.save(file_sigma, sigma)
    sigma2 = np.square(sigma)
    return M, mu, sigma2


def gen_M2nxG(M, weight, sigma2, is_multi=True):
    G = nx.MultiDiGraph() if is_multi else nx.DiGraph()
    for i in range(M.shape[1]):
        start = np.where(M[:, i] == 1)[0].item()
        end = np.where(M[:, i] == -1)[0].item()
        G.add_edge(start, end, mu=weight[i].item(), sigma2=sigma2[i].item(), index=i)
    return G


class MapInfo:
    def __init__(self, net="SiouxFalls", is_multi=True):
        self.M, self.mu, self.sigma2 = gen_M(net)
        self.G = gen_M2nxG(self.M, self.mu, self.sigma2, is_multi)
        self.n_node = self.M.shape[0]
        self.n_edge = self.M.shape[1]

    def get_let_time(self, o, d):
        return nx.dijkstra_path_length(self.G, o-1, d-1, weight='mu')

    def get_let_path(self, o, d):
        return list(np.array(nx.dijkstra_path(self.G, o - 1, d - 1, weight='mu')) + 1)

    def get_ave_time(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            cost = self.G.get_edge_data(path[i] - 1, path[i + 1] - 1)[0]["mu"]
            total_cost += cost
        return total_cost

    def get_sample_time(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            cost = self.get_edge_cost([path[i], path[i + 1]])
            total_cost += cost
        return total_cost

    def get_next_nodes(self, node, zero_mask=False):
        if zero_mask:
            next_nodes = list(map(lambda x: x[1], self.G.edges(node - 1)))
            zero_mask = np.zeros(self.n_node)
            zero_mask[np.array(next_nodes)] = 1
            return zero_mask
        return list(map(lambda x: x[1] + 1, self.G.edges(node - 1)))

    def get_edges(self, node):
        return [[node, node]] + list(map(lambda x: [x[0]+1, x[1]+1], self.G.edges(node-1)))

    def get_edge_weight(self, edge):
        return self.G.get_edge_data(edge[0]-1, edge[1]-1)[0]["mu"]

    def get_edge_cost(self, edge):
        mu = self.G.get_edge_data(edge[0] - 1, edge[1] - 1)[0]["mu"]
        sigma2 = self.G.get_edge_data(edge[0] - 1, edge[1] - 1)[0]["sigma2"]
        cost = np.random.normal(mu, np.sqrt(sigma2))
        return cost

    def get_edge_index(self, edge):
        return self.G.get_edge_data(edge[0]-1, edge[1]-1)[0]["index"]
