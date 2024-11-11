import gurobipy as gp
from gurobipy import GRB
import numpy as np
import func
from func import Map
import time
import pandas as pd
import datetime


def PLM(mymap, S, T, phi=10, e=0.1):
    g_best = -10**7
    g_best_last = -10**7
    probability_last = 0
    max_path = 0
    lmd = np.random.random([S, 1])
    
    samples = func.generate_samples(mymap, S)
    # samples = pd.read_csv('C:/7140209/Weekday_peak_cost.csv').values

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

def ILP(mymap, S, T):

    V = 10**4

    samples = func.generate_samples(mymap, S).T
    # samples = pd.read_csv('C:/7140209/Weekend_off-peak_cost.csv').values.T
    obj_temp1 = np.zeros(mymap.n_link)
    obj_temp2 = np.ones(S)
    obj = np.hstack((obj_temp1, obj_temp2))

    eq_temp = np.zeros([mymap.n_node, S])
    eq_constr = np.hstack((mymap.M, eq_temp))

    ineq_temp = -V * np.eye(S)
    ineq_constr = np.hstack((samples, ineq_temp))

    T = np.ones(S) * T

    n_elem = mymap.n_link + S

    m = gp.Model("ilp")
    m.Params.LogToConsole = 0
    # m.Params.BarConvTol = 1e-2#4
    # m.Params.FeasibilityTol = 1e-2#3
    # m.Params.IntFeasTol = 1e-2#3
    # m.Params.MIPGap = 1e-1#2
    # m.Params.MIPGapAbs = 1e-2#4
    # m.Params.OptimalityTol = 1e-2#4
    # m.Params.MIPFocus = 1
    # m.Params.Presolve = 2

    # z = m.addMVar(shape=n_elem, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    z = m.addMVar(shape=n_elem, vtype=GRB.BINARY, name="z")
    m.setObjective(obj @ z, GRB.MINIMIZE)
    m.addConstr(ineq_constr @ z <= T, name="ineq")
    m.addConstr(eq_constr @ z == mymap.b.reshape(-1), name="eq")
    # m.addConstr(np.eye(n_elem) @ z <= np.ones(n_elem), name="l1")
    # m.addConstr(np.eye(n_elem) @ z >= np.zeros(n_elem), name="g0")
    now = datetime.datetime.now()
    print("开始优化:", f'{now.hour}:{now.minute} {now.second}s\n')
    m.optimize()
    now1 = datetime.datetime.now()
    print("结束优化:", f'{now1.hour}:{now1.minute} {now1.second}s\n')
    print("优化耗时:", f'{now1.hour - now.hour}:{now1.minute - now.minute} {now1.second - now.second}s\n')
    res = z.X
    # print(res)

    prob = 1 - np.dot(obj.T, res).item()/S
    path = np.flatnonzero(res[:mymap.n_link])
    path = func.sort_path_order(path, mymap)
    # print(prob)

    # print("final path:" + str(path + 1))
    return prob, path
