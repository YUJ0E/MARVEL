import numpy as np
import func
import time
import os
from copy import deepcopy
from func import Map, record, write_file
from benchmark import other_iterations
from evaluation import calc_path_prob
from GP4 import GP4_iterations
from cao import ILP, PLM
from prakash import DOT
from yang import MIP_LR, MIP_CPLEX
from fma import FMA_MAP, FMA, LET_path
import math
import datetime

'''
Note: The indexes of nodes and links start from 1 when being set or displayed, but start from 0 when stored and calculated.
'''


def round_nested_list(data, precision=6):
    if isinstance(data, list):
        return [round_nested_list(x, precision) for x in data]
    elif isinstance(data, (int, float)):
        return round(data, precision)
    else:
        return data


np.random.seed(42)


def record(method, res, file_name):
    # res = round_nested_list(res, 6)
    res = [round(x, 6) if isinstance(x, float) else x for x in res]

    fp = open(file_name, 'a+')
    print(method + " prob, g, t, path: " + str(res) + "\n")
    print(time.asctime())
    fp.write(method + " prob, g, t, path: " + str(res) + "\n")
    fp.close()


def write_files(content_name, content, file_list):
    for fl_name in file_list:
        fl = open(fl_name, 'a+')
        fl.write("{}={}\n".format(content_name, content))
        fl.close()


curr_model = "G"  # model can be "G", "log", or "bi"
K = 100  # FMA
S = 100  # PLM ILP OS-MIP
MaxIter = 20  # PLM ILP OS-MIP
DOT_delta = 0.01  # 0.005 #0.01 0.05
PA_maxspeed = 2  # 2

round_num = 4

map_list = ['Anaheim']
# map_list = ['SiouxFalls', 'Friedrichshain', 'Anaheim', 'Winnipeg', 'Chicago_Sketch']
for map_idx in range(0, len(map_list)):
    net = map_list[map_idx]

    curr_dir = os.getcwd()
    t = datetime.datetime.now()
    file_name = os.path.dirname(curr_dir) + f'\\Networks\\{net}\\Benchmark_Record\\OS-MIP_{t.month}_{t.day}_{t.hour}_21_24_bm2.txt'

    file_list = [file_name]
    fp = open(file_name, 'a+')
    fp.write("K={}\n".format(K))
    fp.write("S={}\n".format(S))
    fp.write("MaxIter={}\n".format(MaxIter))
    # fp.write("DOT_delta={}\n".format(DOT_delta))
    fp.write("PA_maxspeed={}\n".format(PA_maxspeed))
    fp.write("\n")
    fp.close()

    # T_factors = np.arange(1, 1.5, 0.05)
    T_factors = [0.95, 1, 1.05]
    kappa = [0.4]

    OD_pairs = func.extract_OD(net)
    OD_pairs = OD_pairs[21:24]

    # t0 = time.perf_counter()
    for i in range(1):
        mymap = FMA_MAP(mapname=net)
        mymap.generate_real_map(net, kappa[i])
        write_file("============", "=============", file_name)
        write_file("kappa=", kappa, file_name)
        write_file("============", "=============", file_name)
        sub_results = {"MLR": []}
        # sub_results = {"DOT": [], "FMA": [], "MLR": [], "ILP": []}
        results = {}
        p_mean = {}
        g_mean = {}
        t_mean = {}
        for tf in T_factors:
            results[tf] = deepcopy(sub_results)
            p_mean[tf] = {}
            g_mean[tf] = {}
            t_mean[tf] = {}

        num = 0
        for OD_idx in range(0, len(OD_pairs)):
            OD = OD_pairs[OD_idx]
            num += 1
            # if num > 10:
            #     break
            write_file("OD", OD, file_name)
            write_file("============", "=============", file_name)
            print("OD={}".format(OD))

            mymap.update_OD(OD)

            for tf in T_factors:
                LET = mymap.dij_cost
                T = tf * LET

                write_file("LET", round(LET, round_num), file_name)
                write_file("tf", round(tf, round_num), file_name)
                write_file("T", round(T, round_num), file_name)

                write_file("=========", "==========", file_name)
                print("========================================================================================")
                print("LET={:.4f},tf={:.4f},T={:.4f}".format(LET, tf, T))

                # # ##############################-----------DOT-----------###################################################
                # DOT_Solver = DOT(mymap, T, DOT_delta)
                # path, cost, g, prob = DOT_Solver.policy2path(T)
                # g_delta = 1 - g
                # res_DOT = prob, g, DOT_Solver.DOT_t_delta, 0, g_delta
                # res_DOT_record = prob, g, DOT_Solver.DOT_t_delta, 0, g_delta, path
                # results[tf]["DOT"].append(res_DOT)
                # record("DOT", res_DOT_record, file_name)

                # ##############################-----------FMA-----------####################################################
                # FMA_Solver = FMA(mymap, T, K)
                # prob, g, t_delta, path = FMA_Solver.policy_iteration()[:4]
                # g_delta = 1 - g
                # res_FMA = prob, g, t_delta, 0, g_delta
                # res_FMA_to_record = prob, g, t_delta, 0, g_delta, path
                # results[tf]["FMA"].append(res_FMA)
                # record("FMA", res_FMA_to_record, file_name)

                # ##############################-----------OS-MIP_LR-----------##############################################
                result = other_iterations(MIP_LR, mymap, T, S, MaxIter)
                res_MLR, node_list = result[:5], result[5]
                res_MLR_to_record = res_MLR, node_list
                results[tf]["MLR"].append(res_MLR)
                record("MLR", res_MLR_to_record, file_name)

                # #############################-----------ILP-----------###########################################
                # res_ILP = other_iterations(ILP, mymap, T, S, MaxIter)
                # results[tf]["ILP"].append(res_ILP)
                # record("ILP", res_ILP, file_name)

                # ##############################-----------PA-----------####################################################
                # prob, g, t_delta = DOT_Solver.PA(T, PA_maxspeed)
                # g_delta = 1 - g
                # res_PA = prob, g, t_delta, 0, g_delta
                # results[tf]["PA"].append(res_PA)
                # record("PA", res_PA, file_name)

                # ##############################-----------PLM-----------####################################################
                # res_PLM = other_iterations(PLM, mymap, T, S, MaxIter)
                # results[tf]["PLM"].append(res_PLM)
                # record("PLM", res_PLM, file_name)

                # # ##############################-----------LET-----------####################################################
                # res_LET = LET_path(mymap, T)
                # results[tf]["LET"].append(res_LET)
                # record("LET", res_LET, file_name)
                # print(time.perf_counter() - t0)
                fp = open(file_name, 'a+')
                fp.write("\n\n")
                fp.close()

        fp = open(file_name, 'a+')
        for tf in results.keys():
            fp.write("tf={:.4f}\n".format(tf))
            for alg in results[tf].keys():
                ret = results[tf][alg]
                p_mean[tf][alg] = np.mean(np.array(ret)[:, 0])
                g_mean[tf][alg] = np.mean(np.array(ret)[:, 1])
                t_mean[tf][alg] = np.mean(np.array(ret)[:, 2])
                ret = [(round(x[0], 4), round(x[1], 6)) for x in ret]
                fp.write(alg + "_results={}\n".format(ret))
            fp.write("\n\n")

        fp.write("p_mean={}\n".format(p_mean))
        fp.write("g_mean={}\n".format(g_mean))
        fp.write("t_mean={}\n".format(t_mean))

        fp.close()
