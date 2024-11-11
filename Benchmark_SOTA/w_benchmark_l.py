from CTD import Trainer, CTD
from utils.func import MapInfo
from utils.cutom_env import *
import func
import os
# from PQL import QL
from GP3 import *
from copy import deepcopy
import time
from ActorCritic import ActorCritic
import datetime
import numpy as np
from hao import *


def write_file(content_name, content, file_name):
    fp = open(file_name, 'a+')
    fp.write("{}={}\n".format(content_name,content))
    fp.close()

def round_nested_list(data, precision=6):
    if isinstance(data, list):
        return [round_nested_list(x, precision) for x in data]
    elif isinstance(data, (int, float)):
        return round(data, precision)
    else:
        return data

def record(method, res, file_name):
    res = round_nested_list(res, 6)
    # res = [round(x, 6) for x in res]
    fp = open(file_name, 'a+')
    print(method + " prob, g, t: " + str(res) + "\n")
    print(time.asctime())
    fp.write(method + " prob, g, t, path: " + str(res) + "\n")
    fp.close()

if __name__ == '__main__':
    T_factors = [1.05, 1, 0.95]

    map_list = ['Winnipeg']
    # map_list = ['SiouxFalls', 'Friedrichshain', 'Anaheim', 'Winnipeg', 'Chicago_Sketch']

    for map_idx in range(0, len(map_list)):
        net = map_list[map_idx]

        map1 = MapInfo(net)
        OD_pairs = func.extract_OD(net)

        t = datetime.datetime.now()
        curr_dir = os.getcwd()
        file_name = os.path.dirname(
            curr_dir) + f'\\Networks\\{net}\\Benchmark_Record\\{t.month}_{t.day}_{t.hour}_bm1.txt'
        sub_results = {"PQL":[]}
        # sub_results = {"GP3":[], "PQL":[], "CTD":[]}


        results = {}
        p_mean = {}
        t_mean = {}
        for tf in T_factors:
            results[tf] = deepcopy(sub_results)
            p_mean[tf] = {}
            t_mean[tf] = {}

        for tf in T_factors:
            write_file("=========", "==========", file_name)
            print("========================================================================================")
            print("tf={}".format(tf))
            for OD in OD_pairs:
                write_file("OD", OD, file_name)
                write_file("============", "=============", file_name)
                print("OD={}".format(OD))
                env1 = Env(map1, OD[0], OD[1])
                LET = map1.get_let_time(OD[0], OD[1])
                T = tf * LET
                write_file("LET", round(LET, 4), file_name)
                write_file("tf", round(tf, 4), file_name)
                write_file("T", round(T, 4), file_name)

                # ---------------------------------GP3------------------------------------##
                # t1_gp3 = time.perf_counter()
                # path, prob = gp3_query(map1, 200, OD, T)
                # # prob_gp3 = calc_path_prob(path, map1, T, samples=None, S=1000)
                # print(prob)
                # t_gp3 = time.perf_counter() - t1_gp3
                # res_gp3 = prob, t_gp3
                # results[tf]["GP3"].append(res_gp3)
                # record("GP3", res_gp3, file_name)

                #---------------------------------PQL------------------------------------##
                pql = PQL(net)
                prob_pql, t_pql, path_list = pql.train(od=OD, budget=tf)
                res_pql = prob_pql, t_pql
                res_pql_to_write = prob_pql, t_pql, path_list
                results[tf]["PQL"].append(res_pql)
                record("PQL", res_pql_to_write, file_name)

                #---------------------------------CTD------------------------------------##
                # ctd = Trainer(env1, policy=CTD, time_budget=T, min_eps=0.1, max_eps=1)
                # ctd.warm_start(1000, lr=0.5)
                # pi_score_ctd = ctd.train(num_train=1000, with_eval=False, int_eval=100)
                # t1_ctd = time.perf_counter()
                # prob_ctd = ctd.eval(1000)[-1]
                # t_ctd = (time.perf_counter() - t1_ctd) / 1000
                # res_ctd = prob_ctd, t_ctd
                # results[tf]["CTD"].append(res_ctd)
                # record("CTD", res_ctd, file_name)

                #---------------------------------AC ------------------------------------##
                # ac = ActorCritic(n_node=env1.map_info.n_node, time_budget=T, learning_rate=0.1, device='cpu', env=env1)
                # # ac.warm_start(1000, lr=0.5)
                # prob_ac, t = ac.train(with_eval=True)
                # res_ac = prob_ac, t
                # results[tf]["AC"].append(res_ac)
                # record("AC", res_ac, file_name)

        fp = open(file_name, 'a+')
        for tf in results.keys():
            fp.write("tf={:.4f}\n".format(tf))
            for alg in results[tf].keys():
                ret = results[tf][alg]
                p_mean[tf][alg] = np.mean(np.array(ret)[:, 0])
                t_mean[tf][alg] = np.mean(np.array(ret)[:, 1])
                ret = [(round(x[0], 4), round(x[1], 6)) for x in ret]
                fp.write(alg + "_results={}\n".format(ret))
            fp.write("\n\n")

        fp.write("p_mean={}\n".format(p_mean))
        fp.write("t_mean={}\n".format(t_mean))

        fp.close()


