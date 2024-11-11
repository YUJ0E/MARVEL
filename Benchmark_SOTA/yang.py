import func
import numpy as np
from func import Map
# from evaluation import calc_path_prob

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
            d_cost, path, x, node_list = func.dijkstra(mymap.G, mymap.r_0, mymap.r_s, ext_weight=samples_tmp)
            phys_cost[w,0] = np.dot(samples[:,w],x)
            # path_prob[w,0] = calc_path_prob(path, mymap, T[0,0], samples)
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

    node_list = [x + 1 for x in node_list]
    print("final path:" + str(np.array(node_list)))
    return up, up_path, node_list

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
    path = func.first_path_link(path, mymap)

    print("final path:" + str(path + 1))
    return prob, path
