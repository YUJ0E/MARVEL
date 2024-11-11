import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    net_name = 'Winnipeg'
    curr_dir = os.getcwd()

    # 定义文件路径
    sigma_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\{net_name}_0.4_random_sigma.npy')
    net_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\{net_name}_network.csv')


    # 你的数组
    arr = np.load(sigma_file)  # 例如，这里是一个简单的数组

    # 读取CSV文件
    df = pd.read_csv(net_file)

    # 确保数组长度与CSV行数一致
    if len(arr) != len(df):
        raise ValueError("数组的长度与CSV文件的行数不一致")

    # 在CSV文件中新增一列'sigma'并写入数组内容
    df['sigma'] = arr

    # 将更新后的DataFrame写回CSV文件
    df.to_csv(net_file, index=False)

    print(f"已成功将数组内容写入CSV文件的'sigma'列")
