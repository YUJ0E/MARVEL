import pandas as pd
import numpy as np
import networkx as nx
import os

class Node:
    def __init__(self, index, parent, nodeId, prob, children, node_type, state, f, status):
        self.index = index  # 节点索引
        self.parent = parent  # 父节点索引
        self.nodeId = nodeId  # 节点 ID
        self.prob = prob  # 节点的概率
        self.children = children  # 子节点列表
        self.type = node_type  # 节点类型（0-OR，1-AND）
        self.state = state  # 节点状态
        self.f = f  # 启发式值
        self.status = status  # 节点状态（True-已解决，False-未解决）

class State:
    def __init__(self, nodeId, actionStatusList):
        self.nodeId = nodeId  # 节点 ID
        self.actionStatusList = actionStatusList  # 动作状态列表

def heuristic_function(nodeId, muGraph, destinationNode):
    c = nx.convert_matrix.from_numpy_array(muGraph)
    try:
        length = nx.shortest_path_length(c, source=nodeId, target=destinationNode, weight='weight')
    except nx.NetworkXNoPath:
        length = float('inf')  # No path found, set heuristic to infinity
    return length

def calculate_assumpt_graph(actionStatusList, muGraph, traversalGraph):
    assumptMuGraph = muGraph.copy()
    assumptTraversalGraph = traversalGraph.copy()
    for action in actionStatusList:
        if action['status'] == 0:
            assumptMuGraph[action['start'], action['end']] = 0
            assumptTraversalGraph[action['start'], action['end']] = 0
        elif action['status'] == 1:
            assumptTraversalGraph[action['start'], action['end']] = 1
    return assumptMuGraph, assumptTraversalGraph

def update_action_list(traversalGraph):
    actionStatusList = []
    rows, cols = traversalGraph.shape
    for i in range(rows):
        for j in range(cols):
            if traversalGraph[i, j] != 0:
                actionStatus = {'start': i, 'end': j, 'status': traversalGraph[i, j]}
                actionStatusList.append(actionStatus)
    return actionStatusList

def expand_children(node, assumptMuGraph, assumptTraversalGraph):
    childrenNodes = []
    parentType = 0
    for i in range(len(assumptTraversalGraph)):
        if 0 < assumptTraversalGraph[node.state.nodeId, i] < 1:
            # Add negative edge child
            childNode_neg = Node(None, node.index, i, 1 - assumptTraversalGraph[node.state.nodeId, i], [], None, State(i, node.state.actionStatusList.copy()), None, False)
            childrenNodes.append(childNode_neg)

            # Add positive edge child
            childNode_pos = Node(None, node.index, i, assumptTraversalGraph[node.state.nodeId, i], [], None, State(i, node.state.actionStatusList.copy()), None, False)
            childrenNodes.append(childNode_pos)

            parentType = 1
        elif assumptTraversalGraph[node.state.nodeId, i] == 1:
            childNode = Node(None, node.index, i, 1, [], None, State(i, node.state.actionStatusList.copy()), None, False)
            childrenNodes.append(childNode)

            parentType = 0
    return childrenNodes, parentType

def add_child_to_tree(childNode, parentIndex, parentType, AOtree):
    childNode.index = len(AOtree)
    AOtree.append(childNode)
    AOtree[parentIndex].children.append(childNode.index)
    AOtree[parentIndex].type = parentType
    return AOtree

def backpropagate(node, AOtree, omega, assumptMuGraph):
    if not node.children:
        return AOtree
    while node.index is not None:
        if AOtree[node.index].type == 0:
            minf = float('inf')
            for childIndex in AOtree[node.index].children:
                AOtree[node.index].status |= AOtree[childIndex].status
                if AOtree[childIndex].f < minf:
                    minf = AOtree[childIndex].f + assumptMuGraph[AOtree[node.index].state.nodeId, AOtree[childIndex].state.nodeId]
            AOtree[node.index].f = minf
        elif AOtree[node.index].type == 1:
            valueExp = 0
            status = True
            for childIndex in AOtree[node.index].children:
                status &= AOtree[childIndex].status
                valueExp += AOtree[childIndex].prob * np.exp(omega * (AOtree[childIndex].f + assumptMuGraph[AOtree[node.index].state.nodeId, AOtree[childIndex].state.nodeId]))
            AOtree[node.index].status = status
            AOtree[node.index].f = (1 / omega) * np.log(valueExp)
        node.index = AOtree[node.index].parent
    return AOtree

def select_min_cost_node(AOtree):
    minf = float('inf')
    nextNodeIndex = 0
    for i, node in enumerate(AOtree):
        if not node.status and node.f < minf:
            minf = node.f
            nextNodeIndex = i
    return AOtree[nextNodeIndex]

def output_path(AOtree, root, destinationNode):
    path = [root.state.nodeId]
    nodeIndex = root.index
    while AOtree[nodeIndex].state.nodeId != destinationNode:
        for childIndex in AOtree[nodeIndex].children:
            if AOtree[childIndex].status:
                path.append(AOtree[childIndex].state.nodeId)
                nodeIndex = childIndex
                break
    return path

def RAO_star(muGraph, sigmaGraph, traversalGraph, currentState, destinationNode, synchronous, all_nodes):
    global path
    path = []

    if currentState == destinationNode:
        return [destinationNode]

    if synchronous == 2:
        return [currentState]

    if synchronous == 1 and path:
        id = path.index(currentState)
        if traversalGraph[currentState][path[id + 1]] == 1:
            return path[id + 1]

    omega = 2
    root = Node(
        index=0,
        parent=None,
        nodeId=currentState,
        prob=1,
        children=[],
        node_type=0,
        state=State(nodeId=currentState, actionStatusList=update_action_list(traversalGraph)),
        f=heuristic_function(currentState, muGraph, destinationNode),
        status=False
    )
    AOtree = [root]
    node = root

    while not AOtree[0].status and AOtree[0].f != float('inf'):
        if len(AOtree) > 10000:
            return [currentState]
        node = select_min_cost_node(AOtree)

        assumptMuGraph, assumptTraversalGraph = calculate_assumpt_graph(node.state.actionStatusList, muGraph, traversalGraph)
        childrenList, parentType = expand_children(node, assumptMuGraph, assumptTraversalGraph)
        for childNode in childrenList:
            assumptMuGraph, assumptTraversalGraph = calculate_assumpt_graph(childNode.state.actionStatusList, muGraph, traversalGraph)
            childNode.f = heuristic_function(childNode.state.nodeId, assumptMuGraph, destinationNode)
            if childNode.state.nodeId == destinationNode:
                childNode.status = True
            AOtree = add_child_to_tree(childNode, node.index, parentType, AOtree)

        if node.nodeId == destinationNode:
            break

        node = select_min_cost_node(AOtree)

        assumptMuGraph, assumptTraversalGraph = calculate_assumpt_graph(node.state.actionStatusList, muGraph, traversalGraph)
        AOtree = backpropagate(node, AOtree, omega, assumptMuGraph)

    path = [all_nodes[node] for node in output_path(AOtree, root, destinationNode)]
    return path

def main():
    # 地图名称，可修改以更改文件路径
    map_name = 'SiouxFalls'

    # 读取CSV文件中的数据
    file_path = f'../Networks/{map_name}/{map_name}_network.csv'
    data = pd.read_csv(file_path)

    # 提取 'From', 'To', 'Cost' 和 'sigma' 列
    from_nodes = data['From'].values
    to_nodes = data['To'].values
    weights = data['Cost'].values
    sigmas = data['Sigma'].values

    # 获取所有节点并创建图的邻接矩阵
    all_nodes = np.unique(np.concatenate((from_nodes, to_nodes)))
    node_count = len(all_nodes)
    node_index = {node: idx for idx, node in enumerate(all_nodes)}

    muGraph = np.zeros((node_count, node_count))
    sigmaGraph = np.zeros((node_count, node_count))
    traversalGraph = np.zeros((node_count, node_count))

    for from_node, to_node, weight, sigma in zip(from_nodes, to_nodes, weights, sigmas):
        muGraph[node_index[from_node], node_index[to_node]] = weight
        sigmaGraph[node_index[from_node], node_index[to_node]] = sigma
        traversalGraph[node_index[from_node], node_index[to_node]] = 1  # 假设遍历图为二进制

    # 读取OD对的CSV文件
    od_file_path = f'../Networks/{map_name}/{map_name}_OD.csv'  # 应该指向OD对的CSV文件
    od_data = pd.read_csv(od_file_path)

    # 获取路由的OD对
    origins = od_data['O'].values
    destinations = od_data['D'].values

    # 打开文件存储路径
    record_folder = f'../Networks/{map_name}/Benchmark_Record'
    os.makedirs(record_folder, exist_ok=True)
    record_file_path = os.path.join(record_folder, 'RAO_Star.txt')

    with open(record_file_path, 'w') as f:
        # 对每一对OD进行运行
        for origin, destination in zip(origins, destinations):
            if origin in node_index and destination in node_index:
                currentState = node_index[origin]
                destinationNode = node_index[destination]
                synchronous = 0

                path = RAO_star(muGraph, sigmaGraph, traversalGraph, currentState, destinationNode, synchronous, all_nodes)
                f.write(f"Path from {origin} to {destination}: {path}\n")
                print(f"Path from {origin} to {destination}: {path}")
            else:
                f.write(f"OD pair ({origin}, {destination}) is out of node index range.\n")
                print(f"OD pair ({origin}, {destination}) is out of node index range.")

if __name__ == "__main__":
    main()
