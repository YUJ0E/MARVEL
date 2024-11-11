import pandas as pd
import networkx as nx
import random
from gensim.models import Word2Vec
import numpy as np
import os

class Node2Vec:
    def __init__(self, data_name, emb_size=128, length_walk=50, num_walks=10, window_size=10, num_iters=5, n=475, repeat_times=900):
        self.data_name = data_name
        self.emb_size = emb_size
        self.length_walk = length_walk
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_iters = num_iters
        self.datasets = {
            'Simple': './Roaddata/Simplifiedroadnet/Simplifiedroadnet.xlsx',
            'Barcelona': './Roaddata/Barcelona/Barcelona-road-network.xlsx',
        }
        self.G = self.load_graph()
        self.n = n
        self.repeat_time = repeat_times

    def load_graph(self):
        # 读取Excel文件
        file_path = self.datasets[self.data_name]
        df = pd.read_excel(file_path)

        # 构建有向图
        G = nx.DiGraph()
        for index, row in df.iterrows():
            from_node = row['From']
            to_node = row['To']
            cost = row['Cost']
            P = row['P']  # 通行概率

            G.add_edge(from_node, to_node, weight=cost, probability=P)
        # 打印网络图信息
        # print(G.edges(data=True))

        return G

    def generate_temp_graph(self, adj_matrix, node_mapping):
        # print('adj_matrix_n:', adj_matrix)
        temp_G = nx.DiGraph()

        for u, v, data in self.G.edges(data=True):
            u2 = node_mapping[u]
            v2 = node_mapping[v]
            if adj_matrix[u2][v2] == 1:
                temp_G.add_edge(u, v, weight=data['weight'])
        # print(temp_G.edges(data=True))
        return temp_G

    def generate_paths(self, temp_G, goal_nodes):
        # print('goal_nodes:', goal_nodes)
        paths = []
        for node in temp_G.nodes():
            for goal_node in goal_nodes:
                if node != goal_node:
                    try:
                        path = nx.dijkstra_path(temp_G, node, goal_node, weight='weight')
                        paths.extend([path] * self.repeat_time)  # 将路径重复100遍
                    except nx.NetworkXNoPath:
                        continue
        return paths

    def random_walk_one_step(self, start_node, temp_G):
        walks = []
        neighbors = list(temp_G.neighbors(start_node))
        if not neighbors:
            return walks # 如果没有邻居，返回n个仅包含起始节点的游走

        weights = [1 / temp_G[start_node][neighbor]['weight'] for neighbor in neighbors]
        probabilities = [w / sum(weights) for w in weights]

        counts = [round(p * self.n) for p in probabilities]  # 使用四舍五入分配给每个邻居节点的步数

        for neighbor, count in zip(neighbors, counts):
            walks.extend([[start_node, neighbor]] * count)
            # print('start_node:', start_node, 'neighbor:', neighbor, 'count:', count)
        return walks

    def generate_random_walks(self, temp_G):
        walks = []
        for node in self.G.nodes():
            walks.extend(self.random_walk_one_step(node, temp_G))
        return walks

    def sentences_from_paths(self, paths):
        sentences = [[str(node) for node in path] for path in paths]
        return sentences


    def train(self, adj_matrix, node_mapping, goal_nodes, workers=4, is_loadmodel=False, is_loaddata=False):
        base_path = './Roaddata/node2vec'
        model_path = f'{base_path}/{self.data_name}.model'
        data_path = f'{base_path}/{self.data_name}.txt'

        os.makedirs(base_path, exist_ok=True)

        if is_loadmodel:
            print('Load model from file')
            w2v = Word2Vec.load(model_path)
            return w2v

        if is_loaddata:
            print('Load data from file')
            with open(data_path, 'r') as f:
                sts = f.read()
                sentences = eval(sts)
        else:
            # print('Generating paths for training data...')
            temp_G = self.generate_temp_graph(adj_matrix, node_mapping)
            paths = self.generate_paths(temp_G, goal_nodes)
            random_walks = self.generate_random_walks(temp_G)
            paths.extend(random_walks)
            # print('paths', paths)
            sentences = self.sentences_from_paths(paths)
            # print('Number of sentences to train: ', len(sentences))
            with open(data_path, 'w') as f:
                f.write(str(sentences))

        print('Start training...')
        random.seed(616)
        w2v = Word2Vec(sentences=sentences, vector_size=self.emb_size, window=self.window_size, sg=1,#Skip-Gram 0-CBOW
                       hs=1, min_count=0, workers=workers, epochs=self.num_iters)
        w2v.save(model_path)
        print('Embedding Done.')

        # node_fir = 3
        # first_node_embedding = w2v.wv[str(node_fir)]
        # print(f'Embedding of the node (3): {first_node_embedding}')
        # print(f'The word at index 3: {w2v.wv.index_to_key[3]}')

        # print("Vocabulary index_to_key list:")
        # for index, word in enumerate(w2v.wv.index_to_key):
        #     print(f"Index {index}: {word}")



        return w2v
