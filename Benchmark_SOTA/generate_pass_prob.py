import csv
import os
import chardet
import numpy as np
import pandas as pd

def write_data(net_name):
    curr_dir = os.getcwd()

    # 定义文件路径
    res_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\{net_name}_network.csv')

    # 读取CSV文件
    df = pd.read_csv(res_file)
    edges = []

    with open(res_file, 'rb') as file:
        content = file.read()
        result = chardet.detect(content)
        encoding = result['encoding']

    # 读取net.csv文件
    with open(res_file, 'r', newline='', encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            edges.append([int(row['From']), int(row['To'])])

    # m是边数
    m = len(edges)
    k = 0.1  # 设置遍历边缘的浓度率

    prob = []
    for i in range(m):
        prob.append(1)

    # 计算遍历边缘数
    num_traversal_edges = int(np.floor(m * k))

    # 随机选择遍历边缘
    all_edges = np.arange(m)
    traversal_edges = np.random.choice(all_edges, num_traversal_edges, replace=False)

    # 生成遍历概率
    traversal_probabilities = np.random.uniform(0.5, 1, num_traversal_edges)

    for i in range(len(traversal_edges)):
        edge_idx = traversal_edges[i]
        prob[edge_idx] = traversal_probabilities[i]

    # 检查是否存在 prob 列
    if 'prob' in df.columns:
        # 如果存在，覆盖该列的内容
        df['prob'] = prob
    else:
        # 如果不存在，新增 sota 列
        df['prob'] = prob

    # 将更新后的内容写回CSV文件，保留其他内容不变
    df.to_csv(res_file, index=False)

    print(f"{net_name}_network.csv prob写入完成")


if __name__ == '__main__':

    # 'SiouxFalls', 'Anaheim', 'Friedrichshain', 'Winnipeg'
    net = 'Winnipeg'
    write_data(net)
