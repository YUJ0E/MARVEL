import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from Embedding_dijkandrandom import Node2Vec  # 导入 Node2Vec 类

class Env:
    def __init__(self, data_name, model_path, emb_size, robots, device):
        self.robots = robots
        self.goal_nodes = [robot['goal_node'] for robot in robots]  # 获取所有机器人终止节点
        self.node2vec = Node2Vec(data_name, emb_size)  # 传入终止节点
        self.num_nodes = self.node2vec.G.number_of_nodes()
        self.node_mapping = {node: idx for idx, node in enumerate(self.node2vec.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_mapping.items()}
        self.device = device
        self.initial_adj_matrix = self.generate_adj_matrix()
        self.adj_matrix = self.initial_adj_matrix.clone().requires_grad_(True).to(device)
        self.original_node_features = self.generate_node_features(model_path, emb_size).to(device)
        self.node_features = self.original_node_features.clone().requires_grad_(True)
        self.new_node_features = self.node_features.clone().requires_grad_(True)
        self.global_time_spent = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.visited_nodes = set()
        self.mean_edge_weights = self.initialize_edge_weights()  # 初始化边的权重

    def generate_adj_matrix(self):
        adj_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float, device=self.device)
        for u, v, data in self.node2vec.G.edges(data=True):
            adj_matrix[self.node_mapping[u], self.node_mapping[v]] = data['probability']
        return adj_matrix

    def generate_node_features(self, model_path, emb_size):
        # Assuming adj_matrix is available in the class
        w2v = self.node2vec.train(self.adj_matrix, self.node_mapping, self.goal_nodes, is_loadmodel=False, is_loaddata=False)
        # first_node_embedding = w2v.wv[3]
        # print(f'Embedding of the node (3): {first_node_embedding}')
        node_features = torch.zeros((self.num_nodes, emb_size + 40), dtype=torch.float, device=self.device)
        for node, idx in self.node_mapping.items():
            # if node == 3:
            #     print('embedding of node 3', w2v.wv[str(node)])
            node_features[idx, :emb_size] = torch.tensor(w2v.wv[str(node)], dtype=torch.float)
        # print('node feature for node 3', node_features[2])

        return node_features

    def update_node_features(self, robot_num, current_node, goal_node):
        self.node_features = self.new_node_features.clone().requires_grad_(True)
        current_node_idx = self.node_mapping[current_node]
        goal_node_idx = self.node_mapping[goal_node]

        # 更新当前节点的倒数第128到第97行
        self.node_features[current_node_idx, -40:-30] = robot_num/10

        # 更新目标节点的倒数第96到第65行
        self.node_features[goal_node_idx, -30:-20] = robot_num/10

        for robot in self.robots:
            if robot['robot_num'] != robot_num:
                other_goal_node_idx = self.node_mapping[robot['goal_node']]
                other_current_node_idx = self.node_mapping[robot['current_node']]

                # 更新其他机器人的目标节点的倒数第64到第33行
                self.node_features[other_current_node_idx, -20:-10] = robot['robot_num']/10

                # 更新其他机器人的当前节点的倒数第32到第1行
                self.node_features[other_goal_node_idx, -10:] = robot['robot_num']/10

    def initialize_edge_weights(self):
        mean_edge_weights = {}
        for u, v, data in self.node2vec.G.edges(data=True):
            mean_cost = torch.tensor(data['weight'], dtype=torch.float, device=self.device)
            sigma = (np.random.uniform(0.2, 0.4) * mean_cost).clone().requires_grad_(True)
            mean_edge_weights[(u, v)] = (mean_cost.clone().requires_grad_(True), sigma.clone().requires_grad_(True))
        return mean_edge_weights

    def sample_edge_cost(self, u, v):
        mean_cost, sigma = self.mean_edge_weights[(u, v)]
        return torch.clamp(torch.normal(mean_cost, sigma), min=mean_cost * 0.7)

    def calculate_total_success_probability(self, T, means, sigmas):
        mu_total = torch.sum(means)
        sigma_total = torch.sqrt(torch.sum(torch.square(sigmas)))
        # print('T', T)
        # print(f"Total mean cost: {mu_total}, total sigma: {sigma_total}")
        z = (T - mu_total) / sigma_total

        S_total = torch.sigmoid(z)
        # print(f"Total success probability: {S_total}")
        return S_total

    def calculate_prob(self, adj_matrix, robots, mean_cost, rob_num):
        probabilities = []
        # 创建临时图
        temp_graph = self.node2vec.G.copy()
        # 根据当前的邻接矩阵设置临时图的边权重
        for u, v, data in temp_graph.edges(data=True):
            if adj_matrix[self.node_mapping[u], self.node_mapping[v]] == 1:
                temp_graph[u][v]['weight'] = data['weight']
            else:
                temp_graph[u][v]['weight'] = float('inf')

        for robot in robots:
            if robot['robot_num'] == rob_num:
                new_time_spent = robot['time_spent'] + mean_cost
            else:
                new_time_spent = robot['time_spent']
            path_cost = torch.tensor(0.0, dtype=torch.float, device=self.device)
            path_sigmas = []
            current_node = robot['current_node']
            goal_node = robot['goal_node']
            try:
                path = nx.dijkstra_path(temp_graph, source=current_node, target=goal_node, weight='weight')
                # print(f"Path from {current_node} to {goal_node}: {path}")
            except nx.NetworkXNoPath:
                # print(f"No path between {current_node} and {goal_node}")
                probabilities.append(torch.tensor(0.0, dtype=torch.float, device=self.device))
                continue
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.node2vec.G.get_edge_data(u, v)
                if edge_data and adj_matrix[self.node_mapping[u], self.node_mapping[v]] == 1:
                    mean_cost, sigma = self.mean_edge_weights[(u, v)]
                    path_cost += mean_cost
                    path_sigmas.append(sigma)
            if len(path_sigmas) > 0:
                path_sigmas = torch.tensor(path_sigmas, dtype=torch.float, device=self.device)
                prob = robot['importance'] * self.calculate_total_success_probability(robot['max_time'] - new_time_spent, path_cost, path_sigmas)
                probabilities.append(prob)
        if not probabilities:
            return torch.tensor(0.0, dtype=torch.float, device=self.device)
        return torch.sum(torch.stack(probabilities))

    def calculate_information_entropy(self, p):
        p_tensor = torch.tensor(p, dtype=torch.float, device=self.device)
        if p > 0:
            return p_tensor * torch.log(p_tensor)
        else:
            return torch.tensor(0.0, dtype=torch.float, device=self.device)

    def update_adj_matrix_and_entropy(self, current_node):
        neighbors = torch.where(self.adj_matrix[current_node] > 0)[0]
        entropies = []
        for neighbor in neighbors:
            prob = self.adj_matrix[current_node, neighbor].item()  # 使用item()获取数值
            if 0 < prob < 1:
                new_prob = 1 if torch.rand(1).item() < prob else 0
                adj_matrix_clone = self.adj_matrix.clone()
                adj_matrix_clone[current_node, neighbor] = new_prob
                entropies.append(self.calculate_information_entropy(prob))
                self.adj_matrix = adj_matrix_clone.clone()


        if not entropies:
            return torch.tensor(0.0, dtype=torch.float, device=self.device, requires_grad=True)
        else:
            return torch.sum(torch.stack(entropies))

    def get_action(self, model, robot, device):
        current_node = robot['current_node']
        goal_node = robot['goal_node']
        rob_num = robot['robot_num']
        valid_edges = self.adj_matrix[self.node_mapping[current_node]] == 1
        next_nodes = torch.where(valid_edges)[0]

        next_node_numbers = [self.idx_to_node[idx.item()] for idx in next_nodes]

        print(f"Next nodes for current node {current_node} (index {self.node_mapping[current_node]})current_agent{robot['robot_num']}: {next_node_numbers}")

        if len(next_nodes) == 0:
            print("No valid edges found!")
            return self.idx_to_node[self.node_mapping[current_node]], [self.idx_to_node[self.node_mapping[current_node]]]  # 返回当前节点和最小cost熵

        with torch.no_grad():

            probs_entropies = []
            original_adj_matrix = self.adj_matrix.clone().requires_grad_(True)
            # print('original_adj_matrix',original_adj_matrix)
            for next_node in next_nodes:
                trans_next_node = self.idx_to_node[next_node.item()]
                adj_matrix_clone = self.adj_matrix.clone().requires_grad_(True)
                if trans_next_node not in self.visited_nodes:
                    entropy = self.update_adj_matrix_and_entropy(next_node)
                else:
                    entropy = torch.tensor(0.0, dtype=torch.float, device=self.device, requires_grad=True)
                # print('adj_matrix',self.adj_matrix,'next_node',trans_next_node)
                edge_data = self.node2vec.G.get_edge_data(current_node, trans_next_node)
                if edge_data:
                    mean_cost, sigma = self.mean_edge_weights[(current_node, trans_next_node)]
                    robot['current_node'] = trans_next_node
                    # print('goal_node',goal_node)
                    if trans_next_node == goal_node:
                        prob_next = 1
                    else:
                        prob_next = self.calculate_prob(self.adj_matrix, self.robots, mean_cost, rob_num)
                    self.adj_matrix = adj_matrix_clone.clone().requires_grad_(True)
                    # print(f"prob: {prob_next} entropy for moving from{trans_next_node}")
                    print('entropy', entropy)
                    print('prob_next', prob_next)
                    prob_entropy = prob_next - entropy
                    print(f"Target: {prob_entropy} for moving to {trans_next_node}")
                    probs_entropies.append(prob_entropy)
                    # print('adj_matrix',self.adj_matrix)
                self.adj_matrix = original_adj_matrix.clone().requires_grad_(True)  # 恢复邻接矩阵
                # print('recovered_adj_matrix',self.adj_matrix)

            robot['current_node'] = current_node

            if not probs_entropies:
                probs_entropies.append(torch.tensor(0.0, dtype=torch.float, device=self.device))

            # print(f"Action probabilities: {action_probabilities}, type: {type(action_probabilities)}")

            max_index = torch.argmax(torch.tensor(probs_entropies))
            # print('max_index',max_index)

            # 使用该索引获取对应的trans_next_node
            max_next_node = next_nodes[max_index]
            print('max_next_node',max_next_node)
            print('max_next_node',self.idx_to_node[max_next_node.item()])
            print('next_nodes',next_nodes)

        return max_next_node, next_nodes

    def custom_cross_entropy_loss(self, predictions, targets):
        epsilon = 1e-12
        total_loss = 0.0
        for i in range(predictions.size(1)):
            total_loss += -targets[0, i] * torch.log(predictions[0, i] + epsilon) - (1 - targets[0, i]) * torch.log(1 - predictions[0, i] + epsilon)
        return total_loss

    def step(self, model, optimizer, criterion, device, model_path, emb_size):
        total_reward = 0
        done = False
        success = False
        loss = 0

        for i, robot in enumerate(self.robots):
            if robot['current_node'] == robot['goal_node']:
                continue

            optimal_next_node, next_nodes = self.get_action(model, robot, device)
            # test_optimal_next_node = optimal_next_node.item()
            # print('optimal_next_node', optimal_next_node, 'idx',self.idx_to_node[test_optimal_next_node])

            from_node_idx = self.node_mapping[robot['current_node']]
            # print('embed_1', self.node_features[2])
            self.update_node_features(robot['robot_num'], robot['current_node'], robot['goal_node'])
            # print('embed_2', self.node_features[2])
            if len(next_nodes) == 1:
                next_node = next_nodes[0].item()
                next_node = self.idx_to_node[next_node]
            else:
                optimizer.zero_grad()
                edge_index = torch.stack(torch.where(self.adj_matrix == 1)).to(device)
                current_node_indices = torch.tensor([from_node_idx], dtype=torch.long, device=device)

                output = model(self.node_features.to(device), edge_index, current_node_indices)
                print('output', output, output.shape)
                print('next_nodes', next_nodes)
                selected_values = output[next_nodes]
                print('selected_values', selected_values)
                q_values_flattened = selected_values.view(1, -1)
                print('q_values_flattened', q_values_flattened)
                q_values_softmax = F.softmax(q_values_flattened, dim=1)
                print('q_values_softmax', q_values_softmax)
                optimal_next_node_index = (next_nodes == optimal_next_node).nonzero(as_tuple=True)[0]
                optimal_next_node_one_hot = torch.zeros(len(next_nodes), device=device)
                optimal_next_node_one_hot[optimal_next_node_index] = 1
                optimal_next_node_one_hot = optimal_next_node_one_hot.view(1, -1)
                print('optimal_next_node_one_hot', optimal_next_node_one_hot)

                # print('optimal_next_node_one_hot', optimal_next_node_one_hot)
                # Compute different loss

                # loss = self.custom_cross_entropy_loss(q_values_softmax, optimal_next_node_one_hot)

                # criterion = nn.MSELoss()
                # loss = 10 * criterion(q_values_softmax, optimal_next_node_one_hot)

                # selected_values = selected_values.unsqueeze(0)
                # optimal_next_node_one_hot = optimal_next_node_one_hot.unsqueeze(0).long()

                criterion = nn.CrossEntropyLoss()
                loss = criterion(q_values_flattened, optimal_next_node_one_hot)

                # criterion = nn.KLDivLoss(reduction='batchmean')
                # loss = criterion(q_values_softmax, optimal_next_node_one_hot)

                print('loss', loss.item())

                softmax_values = F.softmax(selected_values, dim=0)
                softmax_values_cpu = softmax_values.cpu().detach().numpy().flatten()
                print('softmax_values', softmax_values_cpu)
                # print('next_nodes', next_nodes)
                next_edge_index = np.random.choice(len(next_nodes), p=softmax_values_cpu)
                next_node = next_nodes[next_edge_index]

                loss.backward()


                optimizer.step()
                next_node = next_node.item() if isinstance(next_node, torch.Tensor) else next_node
                next_node = self.idx_to_node[next_node]
            to_node_idx = next_node
            edge_data = self.node2vec.G.get_edge_data(self.idx_to_node[from_node_idx], to_node_idx)
            if edge_data is not None:
                time_cost = self.sample_edge_cost(self.idx_to_node[from_node_idx], to_node_idx)
            else:
                time_cost = torch.tensor(100.0, dtype=torch.float, device=self.device, requires_grad=True)

            robot['time_spent'] = robot['time_spent'] + time_cost

            print(f"Robot {robot['robot_num']} starting at {robot['start_node']} moved from {robot['current_node']} to {to_node_idx} with time cost: {time_cost.item()}, total time spent: {robot['time_spent'].item()}")

            if robot['time_spent'].item() > robot['max_time']:
                done = True
                break

            robot['current_node'] = next_node
            # print('adj_matrix', self.adj_matrix)
            if robot['current_node'] not in self.visited_nodes:
                entropy = self.update_adj_matrix_and_entropy(self.node_mapping[robot['current_node']])
                if entropy.item() != 0:
                    # print('changed adj_matrix', self.adj_matrix)
                    self.node_features = self.generate_node_features(model_path, emb_size).to(device)
                    self.new_node_features = self.node_features.clone().requires_grad_(True)
                    # print('changed', self.node_features[2])
                    # print('abcv')
            self.visited_nodes.add(next_node)


        if all(robot['current_node'] == robot['goal_node'] for robot in self.robots):
            success = True
            done = True
        elif any(robot['time_spent'].item() > robot['max_time'] for robot in self.robots):
            done = True

        return total_reward, done, loss, success


    def reset(self):
        self.adj_matrix = self.initial_adj_matrix.clone().requires_grad_(True).to(self.device)
        self.node_features = self.original_node_features.clone().requires_grad_(True).to(self.device)
        self.new_node_features = self.node_features.clone().requires_grad_(True).to(self.device)
        self.visited_nodes = set()
        for robot in self.robots:
            robot['current_node'] = robot['start_node']
            robot['time_spent'] = torch.tensor(0.0, dtype=torch.float, device=self.device).clone().requires_grad_(True)
        # print("Environment reset!")
        # print(f"Node 3 features: {self.node_features[2]}")
