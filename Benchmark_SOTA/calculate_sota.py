import csv
import os
import ast
import chardet
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.stats import norm

matplotlib.use('TkAgg')

def read_data(net_name, alg_name):

    curr_dir = os.getcwd()

    # 定义文件路径
    net_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\{net_name}_network.csv')
    res_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_{alg_name}.csv')

    # 初始化数组
    edges = []
    costs = []
    od_pairs = []
    tf_arr = []
    let_arr = []
    path_arr = []
    prob = []

    with open(net_file, 'rb') as file:
        content = file.read()
        result = chardet.detect(content)
        encoding = result['encoding']

    with open(res_file, 'rb') as file:
        content = file.read()
        result = chardet.detect(content)
        res_encoding = result['encoding']

    # 读取net.csv文件
    with open(net_file, 'r', newline='', encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            edges.append([int(row['From']), int(row['To'])])
            costs.append(float(row['Cost']))
            prob.append(float(row['prob']))

    # 读取res.csv文件
    with open(res_file, 'r', newline='', encoding=res_encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            od_pairs.append(row['od'])
            tf_arr.append(float(row['tf']))
            let_arr.append(float(row['let']))
            path_arr.append(ast.literal_eval(row['path']))

    # 输出数组内容（可以根据需要选择是否需要）
    print("Edges length:", len(edges))
    print("Costs length:", len(costs))
    print("OD_pairs length:", len(od_pairs))
    print("Tf length:", len(tf_arr))
    print("Let length:", len(let_arr))
    print("Path length:", len(path_arr))
    print("prob length:", len(prob))


    sigmas = load_sigma(net_name)

    return edges, costs, od_pairs, tf_arr, let_arr, path_arr, sigmas, prob

def find_idx(two_d_array, one_d_array):

    one_d_array = np.array(one_d_array)
    two_d_array = np.array(two_d_array)

    # 找出二维数组中与一维数组匹配的行索引
    matching_indices = np.where((two_d_array == one_d_array).all(axis=1))[0]

    # 输出结果
    # print("匹配的行索引:", matching_indices)

    return matching_indices

def calculate_miu_sigma(edges, costs, od_pairs, tf_arr, let_arr, path_arr, sigmas, pass_prob):

    real_cost = []
    for i in range(len(od_pairs)):
        real_cost.append([])

    real_sigma = []
    for i in range(len(od_pairs)):
        real_sigma.append([])

    sota = []

    for i in range(len(od_pairs)):
        cur_pass_prob = 1
        od = np.array(ast.literal_eval(od_pairs[i]))
        o, d = od[0], od[1]
        tf = tf_arr[i]
        let = let_arr[i]
        time_budget = tf * let
        path = path_arr[i]

        # 如果没有走到终点
        if path[len(path) - 1] != d:
            time_budget = 0

        if len(path) < 2:
            continue

        #统计real_cost
        for f_idx in range(len(path) - 1):
            t_idx = f_idx + 1
            f = path[f_idx]
            t = path[t_idx]
            edge_idx = find_idx(edges, np.array([f, t]))
            try:
                real_cost[i].append(costs[edge_idx[0]])
                real_sigma[i].append(sigmas[edge_idx[0]] * sigmas[edge_idx[0]])
            except Exception as e:
                print("wrong")
            if pass_prob[edge_idx[0]] < 1:
                cur_pass_prob = cur_pass_prob * pass_prob[edge_idx[0]]

        #均值和、方差和
        mu_sum = np.sum(real_cost[i])
        cov_sum = np.sum(real_sigma[i])

        # 计算小于t的概率即sota
        prob = norm.cdf(time_budget, mu_sum, np.sqrt(cov_sum)) * cur_pass_prob
        # prob = norm.cdf(time_budget, mu_sum, np.sqrt(cov_sum))

        sota.append(prob)

    return sota

def load_sigma(net_name):

    print('-----------------------------load sigma------------------------------')

    curr_dir = os.getcwd()

    # 定义文件路径
    net_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\{net_name}_0.4_random_sigma.npy')

    # 使用 np.load() 加载 .npy 文件
    loaded_data = np.load(net_file)

    # 打印加载的数据
    print("加载的数据:\n", loaded_data)

    # 可以进一步处理 loaded_data，如打印形状、类型等信息
    print("数据形状:", loaded_data.shape)
    print("数据类型:", loaded_data.dtype)

    return loaded_data

def write_data(net_name, alg_name, sota):
    curr_dir = os.getcwd()

    # 定义文件路径
    res_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_{alg_name}.csv')

    # 读取CSV文件
    df = pd.read_csv(res_file)

    # 检查是否存在 sota 列
    if 'sota' in df.columns:
        # 如果存在，覆盖该列的内容
        df['sota'] = sota
    else:
        # 如果不存在，新增 sota 列
        df['sota'] = sota

    # 将更新后的内容写回CSV文件，保留其他内容不变
    df.to_csv(res_file, index=False)

    print(f"{net_name}_{alg_name}.csv sota写入完成")

def prob_mean(net_name, alg_name):

    curr_dir = os.getcwd()

    # 定义文件路径
    res_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_{alg_name}.csv')
    mean_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_{alg_name}_mean_prob.csv')

    # 初始化数组
    tf_arr = []
    prob = []

    with open(res_file, 'rb') as file:
        content = file.read()
        result = chardet.detect(content)
        encoding = result['encoding']

    # 读取res.csv文件
    with open(res_file, 'r', newline='', encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            prob.append(float(row['sota']))
            tf_arr.append(float(row['tf']))

    print("prob length:", len(prob))
    print("Tf length:", len(tf_arr))

    # 创建一个DataFrame
    data = pd.DataFrame({'tf': tf_arr, 'prob': prob})

    # 计算相同tf下prob的均值
    mean_prob = data.groupby('tf')['prob'].mean().reset_index()

    mean_prob.to_csv(mean_file, index=False)

    return mean_prob

if __name__ == '__main__':

    # 'SiouxFalls', 'Anaheim', 'Friedrichshain', 'Winnipeg'
    net = 'Winnipeg'
    # 'DOT', 'FMA', 'OS-MIP', 'PQL', 'DRL'
    alg = 'DRL'
    edges, costs, od_pairs, tf_arr, let_arr, path_arr, sigmas, pass_prob = read_data(net, alg)
    sota = calculate_miu_sigma(edges, costs, od_pairs, tf_arr, let_arr, path_arr, sigmas, pass_prob)
    write_data(net, alg, sota)

    # tf下平均prob
    mean_prob = prob_mean(net, alg)
    print(mean_prob)
