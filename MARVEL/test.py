import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from Embedding_dijkandrandom import Node2Vec
from GAT_zzt_725 import GAT

class Simulation:
    def __init__(self, data_name, emb_size, robots, current_agent, device):
        self.robots = robots
        self.goal_nodes = [robot['goal_node'] for robot in robots]  # 获取所有机器人终止节点
        self.node2vec = Node2Vec(data_name, emb_size)  # 传入终止节点
        self.num_nodes = self.node2vec.G.number_of_nodes()
        self.node_mapping = {node: idx for idx, node in enumerate(self.node2vec.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_mapping.items()}
        self.device = device
        self.initial_adj_matrix = self.generate_adj_matrix()
        self.adj_matrix = self.initial_adj_matrix.clone().requires_grad_(True).to(device)
        self.emb_size = emb_size
        self.node_features = self.generate_node_features().to(device)
        self.visited_nodes = set()
        self.current_agent = current_agent

    def generate_adj_matrix(self):
        adj_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float, device=self.device)
        for u, v, data in self.node2vec.G.edges(data=True):
            adj_matrix[self.node_mapping[u], self.node_mapping[v]] = data['probability']
        return adj_matrix

    def generate_node_features(self):
        w2v = self.node2vec.train(self.adj_matrix, self.node_mapping, self.goal_nodes, is_loadmodel=False, is_loaddata=False)
        node_features = torch.zeros((self.num_nodes, self.emb_size + 40), dtype=torch.float, device=self.device)
        for node, idx in self.node_mapping.items():
            node_features[idx, :self.emb_size] = torch.tensor(w2v.wv[str(node)], dtype=torch.float)
        return node_features

    def update_adj_matrix(self, current_node):
        # 更新邻接矩阵，例如随机移除一些边或更新边的权重
        neighbors = torch.where(self.adj_matrix[current_node] > 0)[0]
        for neighbor in neighbors:
            prob = self.adj_matrix[current_node, neighbor].item()
            if 0 < prob < 1:
                new_prob = 1 if torch.rand(1).item() < prob else 0
                self.adj_matrix[current_node, neighbor] = new_prob

    def run_simulation(self):
        # 设置参数
        data_name = 'Simple_2'
        in_channels = self.emb_size + 40
        hidden_channels = 2 * in_channels
        out_channels = 1
        learning_rate = 1e-5

        # 初始化 GAT 模型
        model = GAT(data_name, learning_rate, in_channels, hidden_channels, out_channels, heads=6).to(self.device)
        # 加载训练好的模型参数
        model_path = './models/model_0830174840_14999.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 将模型设置为评估模式

        robot_num = self.current_agent
        current_node = self.robots[robot_num-1]['current_node']
        goal_node = self.robots[robot_num-1]['goal_node']
        valid_edges = self.adj_matrix[self.node_mapping[current_node]] == 1
        next_nodes = torch.where(valid_edges)[0]

        if len(next_nodes) == 0:
            print('No valid edges for the current node')
            return current_node

        if len(next_nodes) == 1:
            next_node = next_nodes[0].item()
            output_node = self.idx_to_node[next_node]
        else:
            # 更新节点特征矩阵
            self.update_node_features(robot_num, current_node, goal_node)

            # 构造 edge_index，使用 torch.where 获取邻接矩阵中所有为 1 的位置
            edge_index = torch.stack(torch.where(self.adj_matrix == 1)).to(self.device)

            # 选择当前节点索引
            from_node = self.node_mapping[current_node]
            current_node_indices = torch.tensor([from_node], dtype=torch.long, device=self.device)

            # 前向传播，计算模型输出
            with torch.no_grad():
                output = model(self.node_features, edge_index, current_node_indices)
            next_node_outputs = output[next_nodes]

            # 找到最大值及其索引
            max_index = torch.argmax(next_node_outputs)
            selected_next_node = next_nodes[max_index].item()
            output_node = self.idx_to_node[selected_next_node]

        # 更新邻接矩阵并重新嵌入特征
        self.update_adj_matrix(self.node_mapping[current_node])
        self.node_features = self.generate_node_features().to(self.device)

        print(f"机器人 {robot_num} 当前节点: {current_node}, 下一节点: {output_node}")
        return output_node

if __name__ == '__main__':
    # 设置设备为 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 机器人信息，包含每个机器人的编号、当前节点和目标节点
    robots = [
        {'robot_num': 1, 'current_node': 9, 'next_node': 9, 'goal_node': 12},
        {'robot_num': 2, 'current_node': 1, 'next_node': 1, 'goal_node': 8}
    ]

    # 模型路径
    model_path = './models/model_0830174840_14999.pth'
    while not all(robot['current_node'] == robot['goal_node'] for robot in robots):
        # 对每个机器人运行仿真
        for robot in robots:
            current_agent = robot['robot_num']
            if robot['current_node'] != robot['goal_node']:
                sim = Simulation(data_name='Simple_2', emb_size=128, robots=robots, current_agent=current_agent, device=device)

                # 运行仿真
                output_node = sim.run_simulation()
                robots[current_agent - 1]['current_node'] = output_node

    print('所有机器人都已到达目标节点')
