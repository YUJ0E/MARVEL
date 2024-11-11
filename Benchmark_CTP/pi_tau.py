from importlib.abc import Traversable
import numpy as np
import networkx as nx
import pandas as pd
import random
from network_operations import get_travel_time
from scipy.stats import norm
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def create_network(edges_data):
    G = nx.DiGraph()
    
    for _, edge in edges_data.iterrows():
        #print(edge)
        if edge['prob']>=1:
            G.add_edge(int(edge['From']), int(edge['To']), weight=edge['Cost'], sigma=random.uniform(0,0.4)*edge['Cost'],
                    p=edge['prob'], edge_type='edge1')
            
        if edge['prob'] < 1:
            G.add_edge(int(edge['From']), int(edge['To']), weight=edge['Cost'], sigma=random.uniform(0,0.4)*edge['Cost'],
                       p=edge['prob'], edge_type='edge2')
    
        
    return G

map_name='Winnipeg' # Options: SiouxFalls, Anaheim, Friedrichshain, Winnipeg

datas=load_data(f'../Networks/{map_name}/{map_name}_network.csv')
G=create_network(datas)
tau=0.7
G2=G
edges = list(G.edges(data=True))
for u,v,data in edges:
    if data['p']<tau:
        G2.remove_edge(u,v)

times=100
ods=load_data(f'../Networks/{map_name}/{map_name}_OD.csv')
paths=[]
datas=[]

meandata=[]
m1=pd.Series()
m1['tf']=1
m2=pd.Series()
m2['tf']=1.05
m3=pd.Series()
m3['tf']=0.95

p=[]
p1=pd.Series()
p1['tf']=1
p2=pd.Series()
p2['tf']=1.05
p3=pd.Series()
p3['tf']=0.95

path_num=0
p_1=p_2=p_3=0
for _, od in ods.iterrows():
    
    c1=c2=c3=0
    l=pd.Series()
    l_105=pd.Series()
    l_095=pd.Series()
    no_path_flag=False
    try:
        shortest_path = nx.shortest_path(G2, od['O'], od['D'], weight='weight')
        shortest_cost = sum(G[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
        shortest_path_sigma=sum(G[u][v]['sigma'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    except Exception:
        print(Exception)
        shortest_cost='NaN'
        no_path_flag=True

    for _ in range(times):
        path_output=pd.Series()
        path_output['od']=(od['O'],od['D'])
        G_tmp=G2
        flag=True
        while flag:
            path=[od['O']]
            try:
                path=nx.shortest_path(G_tmp,od['O'],od['D'])
            except Exception:
                flag=False
                print(Exception)
            if len(path)>=2:
                for i in range(len(path)-1):
                    if random.random()>G_tmp[path[i]][path[i+1]]['p']:
                        G_tmp.remove_edge(path[i],path[i+1])
                        break
                    if i==len(path)-2:
                        flag=False
        path_output['path']=path
        mean=sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        if len(path)>=2:        
            path_output['mean']=mean
            sigma=sum(G[u][v]['sigma'] for u, v in zip(path[:-1], path[1:]))
            path_output['sigma']=sigma
            path_output['LET']=shortest_cost
            cost=sum(get_travel_time(G[u][v]['weight'], G[u][v]['sigma']) 
                    for u, v in zip(path[:-1], path[1:]))
            c1+=norm.cdf(shortest_cost,mean,sigma)
            c2+=norm.cdf(shortest_cost*1.05,mean,sigma)
            c3+=norm.cdf(shortest_cost*0.95,mean,sigma)
            paths.append(path_output)

    if not no_path_flag:
        path_num+=1
        p_1+=c1/times
        p_2+=c2/times
        p_3+=c3/times
        l['od']=l_095['od']=l_105['od']=(od['O'],od['D'])
        l['tf']=1
        l_105['tf']=1.05
        l_095['tf']=0.95
        l['let']=l_095['let']=l_105['let']=shortest_cost
        l['prob']=norm.cdf(shortest_cost,shortest_cost,shortest_path_sigma)
        l_105['prob']=norm.cdf(shortest_cost*1.05,shortest_cost,shortest_path_sigma)
        l_095['prob']=norm.cdf(shortest_cost*0.95,shortest_cost,shortest_path_sigma)
        l['sota']=c1/times
        l_105['sota']=c2/times
        l_095['sota']=c3/times
        datas.append(l)
        datas.append(l_105)
        datas.append(l_095)
p1['prob']=p_1/path_num
p2['prob']=p_2/path_num
p3['prob']=p_3/path_num
p.append(p1)
p.append(p2)
p.append(p3)

output_path = pd.DataFrame(paths) if len(paths) > 0 else pd.DataFrame()
# Ensure directory exists before saving files
output_dir = f'../Networks/{map_name}/Benchmark_Record'
os.makedirs(output_dir, exist_ok=True)
output_path.to_csv(f'../Networks/{map_name}/Benchmark_Record/pi_tau.csv', index=False)