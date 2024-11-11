import random

from numpy import NaN
from pandas import isnull
from network_operations import create_network,get_travel_time
from imports import *
from simulated_annealing import simulated_annealing

test_time=1000

paths=pd.read_csv('data\Winnipeg_paths.csv')
network=pd.read_csv('experiment\Winnipeg\Winnipeg_network_0.1_rand.csv')
G=create_network(network)
new_paths=[]
for _, path in paths.iterrows():
    p=[]
    for item in path:
        if not isnull(item):
            p.append(int(item))
        else:
            break
    print(p)
    new_paths.append(p)
datas=[]

for path in new_paths:
    l=pd.Series()
    l_105=pd.Series()
    l_095=pd.Series()
    if len(path)!=1:
        shortest_path = nx.shortest_path(G, path[0], path[len(path)-1], weight='weight')
        shortest_cost = sum(G[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
        c1=c2=c3=0
        for _ in range(test_time):
            cost=0
            cost=sum(get_travel_time(G[u][v]['weight'], G[u][v]['sigma']) 
                     for u, v in zip(path[:-1], path[1:]))
            if cost<shortest_cost:
                c1+=1
            if cost<shortest_cost*1.05:
                c2+=1
            if cost<shortest_cost*0.95:
                c3+=1

        l['od']=l_095['od']=l_105['od']=(path[0],path[len(path)-1])
        l['LET']=l_095['LET']=l_105['LET']=shortest_cost
        l['path']=l_095['path']=l_105['path']=path
        l['tf']=1
        l_105['tf']=1.05
        l_095['tf']=0.95
        l['prob']=c1/test_time
        l_105['prob']=c2/test_time
        l_095['prob']=c3/test_time
        datas.append(l)
        datas.append(l_105)
        datas.append(l_095)

d=pd.DataFrame(datas)
d.to_csv('winnipeg_result.csv',index=False)