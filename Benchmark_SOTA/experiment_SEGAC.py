import torch
import EVFA
import GPG
from utils.cutom_env import *
from trainer import GeneralizedAC
from tqdm import tqdm
import func
from func import Map, record, write_file
import os
# from fma import FMA_MAP, FMA, LET_path
import time


if __name__ == '__main__':
    random.seed(808)
    np.random.seed(808)
    torch.manual_seed(808)
    torch.cuda.manual_seed(808)
    curr_dir = os.getcwd()
    map_dir = curr_dir + '/nn/'
    map_id = 2#map_id can be integers from 0~7
                # 0    Sioux Falls
                # 1    Anaheim
                # 2    Winnipeg
                # 3    Chicago-Sketch
                # 4    Chengdu-Weekend Off-peak Hour
                # 5    Chengdu-Weekend Peak Hour  
                # 6    Chengdu-Weekday Off-peak Hour
                # 7    Chengdu-Weekday Peak Hour
    map1 = MapInfo("./nn/Winnipeg/Winnipeg_network.csv")
    # map1 = MapInfo("D:/SE-GAC-master/nn/Anaheim/Anaheim_network.csv")
    # map1 = MapInfo("D:/GE-GAC/GE-GAC/nn/Chicago_Sketch/Chicago_Sketch_network. csv")
    # mymap = FMA_MAP()
    # mymap = Map()
    # mymap.generate_real_map(map_id, map_dir)
    # OD_pairs = [[1,87]]
    OD_pairs = func.extract_OD(map_id, map_dir)
    # T_factors = [0.95, 1, 1.05]                                                                                                   
    T_factors = [1.05, 1, 0.95]
    res = [[] for _ in range(3)]
    index = 0
    res_t = [[] for _ in range(3)]
    for tf in T_factors: 
        for OD in OD_pairs:
            path = str(OD[0]) + '-' + str(OD[1]) + '_episode=99' + '.pth'
            env1 = Env(map1, OD[0], OD[1])
            T = tf * map1.get_let_time(OD[0], OD[1])
            GPG.WITH_WIS = False
            EVFA.WITH_VARCON = False
            # gac = GeneralizedAC(env1, time_budget=T, buffer_size=100, mode='on-policy', with_critic=False, device='cuda', ckpt=path)
            gac = GeneralizedAC(env1, time_budget=T, buffer_size=100, mode='on-policy', with_critic=False, device='cpu')
            # gac.supervised_warm_start(10000, destination_node=OD[1], save=False)
            gac.warm_start(10, epsilon=0.2, start_node=OD[0], destination_node=OD[1], save=False) 
            # print(gac.eval(1,1000))
            # pi_score = gac.train(num_train=100, batch_size=100, with_eval=False, int_eval=5, start_node=OD[0], destination_node=OD[1], save=True)         
            # res[index].append(gac.eval(1,1000))
            t1 = time.perf_counter()
            res[index].append(gac.eval(1,1000))
            t = time.perf_counter() - t1
            res_t[index].append(t)
            print(res)
        index += 1
    print('---------------------------------final prob---------------------------------')
    print(res)
    print('---------------------------------final time---------------------------------')
    print(res_t)
    print('---------------------------------1.05---------------------------------')
    print(np.mean(res[0]))
    print(np.mean(res_t[0]))
    print('---------------------------------1---------------------------------')
    print(np.mean(res[1]))
    print(np.mean(res_t[1]))
    print('---------------------------------0.95---------------------------------')
    print(np.mean(res[2]))
    print(np.mean(res_t[2]))

    