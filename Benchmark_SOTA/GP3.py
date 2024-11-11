import networkx as nx
import numpy as np
from utils.cutom_env import *
from scipy.stats import norm


def weight_func(k):
    return lambda u, v, d: min(k * attr.get("mu", 1) + np.sqrt(attr.get("sigma2", 1)) for attr in d.values())


def get_path_mu_sigma2(mymap, path):
    mu = 0
    sigma2 = 0
    for i in range(len(path) - 1):
        mu += mymap.G.get_edge_data(path[i], path[i+1])[0]["mu"]
        sigma2 += mymap.G.get_edge_data(path[i], path[i + 1])[0]["sigma2"]
    return mu, sigma2

def generate_samples(mymap, S):
    var = mymap.sigma2.flatten()
    cov = np.diag(var)
    rng = np.random.default_rng()
    samples = rng.multivariate_normal(mymap.mu.reshape(-1), cov, S, method='cholesky')
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            while samples[i][j] <= 0:
                samples[i][j] = np.random.normal(mymap.mu[j].item(), np.sqrt(mymap.sigma2[j]))
    return samples.T

def calc_path_prob(path, mymap, T, samples=None, S=1000):    
    samples = generate_samples(mymap, S)
    x = np.zeros(mymap.n_edge)
    x[path-1] = 1
    return np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]


def gp3_query(mymap, k, OD, T):
    p = nx.dijkstra_path(G=mymap.G, source=OD[0]-1, target=OD[1]-1, weight='mu')
    cur_mu, cur_sigma2 = get_path_mu_sigma2(mymap, p)
    cur_sigma = np.sqrt(cur_sigma2)
    best_prob = norm(cur_mu, cur_sigma).cdf(T)
    for i in range(k):
        q = nx.dijkstra_path(mymap.G, source=OD[0]-1, target=OD[1]-1, weight=weight_func(k=i))
        cur_mu, cur_sigma2 = get_path_mu_sigma2(mymap, q)
        cur_sigma = np.sqrt(cur_sigma2)
        cur_prob = norm(cur_mu, cur_sigma).cdf(T)
        if cur_prob > best_prob:
            best_prob = cur_prob
            p = q
    return np.array(p) + 1, best_prob


if __name__ == '__main__':
    map1 = MapInfo("maps/sioux_network.csv")
    path, prob = gp3_query(map1, 100, [1, 15], 43)
    print(path, prob)
    print(calc_path_prob(path, map1, 43, samples=None, S=1000))


