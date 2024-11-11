from _CTD import Trainer, CTD
from utils.cutom_env import *
import func
import os
from PQL import QL
from GP3 import *
from copy import deepcopy
import time
from ActorCritic import ActorCritic

map1 = MapInfo("D:/SE-GAC-master/nn/Anaheim/Anaheim_network.csv")
print(map1.G.edges)