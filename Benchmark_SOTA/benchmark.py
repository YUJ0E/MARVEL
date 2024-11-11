import numpy as np
import func
import time
from evaluation import calc_path_prob
from scipy.stats import norm
from fma import calc_path_g

def DOT(mymap, T, delta=1):
    n_timestamp = np.ceil(T/delta).astype(int)
    J = np.zeros([mymap.n_node, n_timestamp+1])
    J[mymap.r_s,:] = 1
    U = -1 * np.ones([mymap.n_node, n_timestamp+1])
    times = np.arange(0, T, delta).reshape(-1,1)

    if mymap.model == 'G':
        CDF = norm.cdf(times, mymap.mu.reshape(-1), np.sqrt(np.diag(mymap.cov)))
        CDF_delta = CDF[1:, :] - CDF[:n_timestamp-1, :]
    elif mymap.model == 'log':
        CDF = norm.cdf(np.log(times), mymap.mu.reshape(-1), np.sqrt(np.diag(mymap.cov)))
        CDF_delta = CDF[1:, :] - CDF[:n_timestamp-1, :]
    elif mymap.model == 'bi':
        CDF1 = norm.cdf(times, mymap.mu.reshape(-1), np.sqrt(np.diag(mymap.cov)))
        CDF1_delta = CDF1[1:, :] - CDF1[:n_timestamp-1, :]
        CDF2 = norm.cdf(times, mymap.mu2.reshape(-1), np.sqrt(np.diag(mymap.cov2)))
        CDF2_delta = CDF2[1:, :] - CDF2[:n_timestamp-1, :]
        CDF_delta = func.calc_bi_gauss(mymap.phi_bi, CDF1_delta, CDF2_delta)

    for timestamp in range(n_timestamp-1,-1,-1):
        t = times[timestamp]
        for node in mymap.G.nodes:
            if node != mymap.r_s:
                prob_max = 0
                u = -1
                for _, next_node, d in mymap.G.out_edges(node, data=True):
                    link_idx = d['index']
                    prob = 0
                    
                    prob += np.dot(CDF_delta[:n_timestamp-timestamp-1, link_idx], J[next_node, timestamp+1:n_timestamp])

                    t_upper = T - t
                    timestamp_lower = n_timestamp-timestamp-1
                    if mymap.model == 'G':
                        prob_tprime = norm.cdf(t_upper, mymap.mu[link_idx], np.sqrt(mymap.cov[link_idx, link_idx]))\
                                    - CDF[timestamp_lower, link_idx]
                    elif mymap.model == 'log':
                        prob_tprime = norm.cdf(np.log(t_upper), mymap.mu[link_idx], np.sqrt(mymap.cov[link_idx, link_idx]))\
                                    - CDF[timestamp_lower, link_idx]
                    elif mymap.model == 'bi':
                        prob_temp1  = norm.cdf(t_upper, mymap.mu[link_idx], np.sqrt(mymap.cov[link_idx, link_idx]))\
                                    - CDF1[timestamp_lower, link_idx]
                        prob_temp2  = norm.cdf(t_upper, mymap.mu2[link_idx], np.sqrt(mymap.cov2[link_idx, link_idx]))\
                                    - CDF2[timestamp_lower, link_idx]
                        prob_tprime = func.calc_bi_gauss(mymap.phi_bi, prob_temp1, prob_temp2)
                    prob += prob_tprime * J[next_node, n_timestamp]

                    if prob >= prob_max:
                        prob_max = prob
                        u = link_idx
                
                J[node, timestamp] = prob_max
                U[node, timestamp] = u
    
    return J[mymap.r_0, 0].item(), J, U.astype(int)


def PLM(mymap, S, T, phi=10, e=0.1):
    g_best = -10**7
    g_best_last = -10**7
    probability_last = 0
    max_path = 0
    lmd = np.random.random([S, 1])
    
    samples = func.generate_samples(mymap, S)
    
    T = np.ones([S, 1]) * T

    k = 1
    k_x = 0

    while(True):
        d_cost, path, x = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s, ext_weight=np.dot(samples, lmd))
        sub1_cost = d_cost - np.dot(T.T, lmd)

        tmp = np.ones([S, 1])-lmd
        xi = np.where(tmp > 0, 0, 10**7)
        sub2_cost = np.sum(np.dot(tmp.T, xi))

        cost = sub1_cost + sub2_cost
        probability = np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]

        if probability >= probability_last:
            max_path = path
            probability_last = probability
        probability = max(probability, probability_last)
        # print(k)
        # print(cost)
        # print(max_path)

        g_best = max(cost, g_best)
        if (g_best - g_best_last >= e):
            k_x = k
        g_best_last = g_best

        if(k-k_x >= phi):
            break

        d_g = np.dot(samples.T, x) - T - xi

        alpha = 0.0001/np.sqrt(k)
        lmd += alpha * d_g
        lmd = np.where(lmd > 0, lmd, 0)
        
        k += 1

    # print("final path:" + str(np.array(max_path) + 1))
    return probability, max_path

def MIP_LR(mymap, S, T, phi=5, e=1):
    g_best = -10**7
    g_best_last = -10**7
    up_last = 0
    up_path = 0
    M = 10**4
    rho = np.random.random()
    lmd = np.random.random([S, 1])

    samples = func.generate_samples(mymap, S)

    T = np.ones([S, 1]) * T

    k = 1
    k_x = 0

    while(True):
        sigma = 1 if 1-rho >= 0 else 0
        sub1_cost = min(0, 1-rho)

        tmp = -M*lmd + rho/S
        z_w = np.where(tmp >= 0, 0, 1)
        sub2_cost = np.dot(tmp.T, z_w)

        paths = []
        d_cost_total = 0
        
        phys_cost = np.zeros([S, 1])
        path_prob = np.zeros([S, 1])
        for w in range(S):
            samples_tmp = lmd[w,0]*samples[:,w].reshape(-1,1)
            d_cost, path, x = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s, ext_weight=samples_tmp)
            phys_cost[w,0] = np.dot(samples[:,w],x)
            path_prob[w,0] = np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0))/S
            paths.append(path)
            d_cost_total += d_cost

        sub3_cost = d_cost_total - np.dot(T.T, lmd)

        cost = sub1_cost + sub2_cost + sub3_cost
        if np.max(path_prob) >= up_last:
            up = np.max(path_prob)
            up_path = paths[np.argmax(path_prob)]
            up_last = up
        # print(k)
        # print(cost)
        # print(up_path)
        
        probability = 1 - np.sum(z_w)/S

        g_best = max(cost, g_best)
        if (g_best - g_best_last >= e):
            k_x = k
        g_best_last = g_best

        if(k-k_x >= phi):
            break
        
        d_rho = sigma - probability
        d_lmd = -M*z_w - T + phys_cost

        alpha = 0.00001 / np.sqrt(k)
        rho += alpha * d_rho
        rho = max(0, rho)
        lmd += alpha * d_lmd
        lmd = np.where(lmd > 0, lmd, 0)

        k += 1

    # print("final path:" + str(np.array(up_path) + 1))
    return up, up_path

def MIP_CPLEX(mymap, S, T):
    M = 10*3

    samples = func.generate_samples(mymap, S).T

    obj_temp1 = np.zeros([mymap.n_link,1])
    obj_temp2 = np.ones([S, 1])
    obj = np.vstack((obj_temp1, obj_temp2))

    eq_temp = np.zeros([mymap.n_node, S])
    eq_constraint = np.hstack((mymap.M, eq_temp))

    ineq_temp = -M * np.eye(S)
    ineq_constraint = np.hstack((samples, ineq_temp))

    T = np.ones([S, 1]) * T

    res = func.cvxopt_glpk_binary(obj, ineq_constraint, T, eq_constraint, mymap.b)
    prob = 1 - np.dot(obj.T, res).item()/S
    path = np.flatnonzero(res[:mymap.n_link])

    # print("final path:" + str(path + 1))
    return prob, path

def other_iterations(alg, mymap, T, S, MaxIter):
    '''
    run a certain algorithm 'alg' for MaxIter times and return the statistics
    '''

    pro = []
    g = []
    t_delta = []
    g_delta = []
    for ite in range(MaxIter):
        print('{} iteration #{}'.format(alg.__name__, ite))
        t1 = time.perf_counter()
        _, path, node_list = alg(mymap, S, T)
        t_delta.append(time.perf_counter() - t1)
        print("final path: {}\n".format(str(np.array(path) + 1)))
        pro.append(calc_path_prob(path, mymap, T))
        g.append(calc_path_g(path, mymap, T))
        g_delta.append(1-calc_path_g(path, mymap, T))
    return np.mean(pro), np.mean(g), np.mean(t_delta), np.max(t_delta), np.mean(g_delta), node_list
