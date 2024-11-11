# 相同的OD读入LET
import pandas as pd
import os

def handle_data(net_name):


    curr_dir = os.getcwd()

    # 定义文件路径
    dot_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_DOT.csv')
    drl_file = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_DRL.csv')

    # 读取 file1 和 file2
    file1 = pd.read_csv(dot_file)
    file2 = pd.read_csv(drl_file)

    # 创建一个字典，用于快速查找 file1 中的 od 和 let 对应关系
    od_let_dict = dict(zip(file1['od'], file1['let']))

    # 定义一个函数来根据 od 值更新 let 列
    def update_let(row):
        if row['od'] in od_let_dict:
            return od_let_dict[row['od']]
        return row['let']

    # 使用 apply 方法更新 file2 中的 let 列
    file2['let'] = file2.apply(update_let, axis=1)

    # 将更新后的 file2 写入新的 CSV 文件
    file2.to_csv(drl_file, index=False)
    print("更新完成")

if __name__ == '__main__':
    handle_data('Anaheim')
