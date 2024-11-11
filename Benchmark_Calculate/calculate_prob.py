import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

map_name = 'Simple'  # Options: SiouxFalls, Anaheim, Friedrichshain, Winnipeg
file_path = f'../Networks/{map_name}/{map_name}_network.csv'  # 请替换成你的文件路径
df = pd.read_csv(file_path)

# 提取相关列
edge_data = df[['From', 'To', 'Cost', 'sigma', 'prob']]

# 定义采样边的时间消耗函数，使用numpy生成正态分布的随机数
def sample_edge_cost(mean_cost, sigma):
    sampled_cost = np.random.normal(mean_cost, sigma)
    # 限制采样时间不小于均值的70%
    return max(sampled_cost, mean_cost * 0.6)

# 定义计算路径中所有边的通行概率乘积的函数
def calculate_total_pass_prob(path):
    total_pass_prob = 1  # 用于存储通行概率的乘积

    # 遍历路径中的每一条边（相邻点成对处理）
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge = edge_data[(edge_data['From'] == u) & (edge_data['To'] == v)].iloc[0]
        pass_prob = edge['prob']  # 获取边的通行概率

        # 计算通行概率的乘积
        total_pass_prob *= pass_prob

    return total_pass_prob

# 定义单次模拟按时到达的函数
def simulate_single_iteration(path, target_time):
    total_time = 0

    # 遍历路径中的每一条边（相邻点成对处理）
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge = edge_data[(edge_data['From'] == u) & (edge_data['To'] == v)].iloc[0]
        mean_cost = edge['Cost']
        sigma = edge['sigma']

        # 采样当前边的时间
        edge_time = sample_edge_cost(mean_cost, sigma)
        total_time += edge_time

    # 判断总时间是否在目标时间以内
    return total_time <= target_time

# 定义并行模拟按时到达概率的函数
def simulate_on_time_arrival(path, target_time, num_iterations):
    on_time_count = 0

    # 使用 ThreadPoolExecutor 实现并行模拟
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda _: simulate_single_iteration(path, target_time), range(num_iterations)))

    # 统计按时到达的次数
    on_time_count = sum(results)

    # 计算按时到达的概率
    on_time_prob = on_time_count / num_iterations
    return on_time_prob

# 主程序
def main():
    # 定义路径
    path = [9, 10.0, 13, 11.0, 12]

    # 输入预算时间
    budget_time = 4

    # 输入模式
    mode = 'main'

    # 根据模式设置目标时间
    if mode == 'main':
        target_times = [0.95 * budget_time, budget_time, 1.05 * budget_time]
    elif mode == 'others':
        target_times = [1.2 * budget_time]
    else:
        print("无效的模式输入，请输入 'main' 或 'others'")
        return

    # 计算路径中每条边的通行概率乘积
    total_pass_prob = calculate_total_pass_prob(path)

    # 运行模拟并输出结果
    num_iterations = 10000
    for target_time in target_times:
        on_time_probability = simulate_on_time_arrival(path, target_time, num_iterations)
        final_probability = on_time_probability * total_pass_prob
        print(f"目标时间 {target_time:.2f} 下的最终按时到达的概率（乘以通行概率）: {final_probability:.4f}")

if __name__ == "__main__":
    main()
