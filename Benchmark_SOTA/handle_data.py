import csv
import os
import re


def handle_dot_data(net_name, txt_name):
    # 读取文件内容
    with open(txt_name, 'r') as file:
        data = file.read()

    # 正则表达式模式
    OD_pattern = re.compile(r'OD=\[([0-9,\s]+)\]')
    LET_pattern = re.compile(r'LET=([\d.]+)')
    tf_pattern = re.compile(r'tf=([\d.]+)')
    prob_pattern = re.compile(r'DOT prob, g, t, path: \[([\d.]+)')
    path_pattern = re.compile(r'DOT prob, g, t, path: \[[\d.,\s]+, \[([0-9,\s]+)\]\]')

    # 匹配数据
    OD_matches = OD_pattern.findall(data)
    LET_matches = LET_pattern.findall(data)
    tf_matches = tf_pattern.findall(data)
    prob_matches = prob_pattern.findall(data)
    path_matches = path_pattern.findall(data)

    # 使用列表推导式将每个OD元素重复三次
    OD_matches = [od for od in OD_matches for _ in range(3)]

    # 确保所有匹配的列表长度相同
    min_length = min(len(OD_matches), len(LET_matches), len(tf_matches), len(prob_matches), len(path_matches))

    # 数据文件名
    curr_dir = os.getcwd()
    filename = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_DOT.csv')

    # 表头
    header = ['od', 'tf', 'let', 'prob', 'path']

    # 创建并写入CSV文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(header)

        # 写入数据
        for i in range(min_length):
            od = [int(x) for x in OD_matches[i].split(',')]
            let = float(LET_matches[i])
            tf = float(tf_matches[i])
            prob = float(prob_matches[i])
            path = [int(x) for x in path_matches[i].split(',')]

            writer.writerow([od, tf, let, prob, path])

    print(f"CSV文件 '{filename}' 已创建并写入数据.")

def handle_fma_data(net_name, txt_name):
    # 读取文件内容
    with open(txt_name, 'r') as file:
        data = file.read()

    # 正则表达式模式
    OD_pattern = re.compile(r'OD=\[([0-9,\s]+)\]')
    LET_pattern = re.compile(r'LET=([\d.]+)')
    tf_pattern = re.compile(r'tf=([\d.]+)')
    prob_pattern = re.compile(r'FMA prob, g, t, path: \[([\d.]+)')
    path_pattern = re.compile(r'FMA prob, g, t, path: \[[\d.,\s]+, \[([0-9,\s]+)\]\]')

    # 匹配数据
    OD_matches = OD_pattern.findall(data)
    LET_matches = LET_pattern.findall(data)
    tf_matches = tf_pattern.findall(data)
    prob_matches = prob_pattern.findall(data)
    path_matches = path_pattern.findall(data)

    # 使用列表推导式将每个OD元素重复三次
    OD_matches = [od for od in OD_matches for _ in range(3)]

    # 确保所有匹配的列表长度相同
    min_length = min(len(OD_matches), len(LET_matches), len(tf_matches), len(prob_matches), len(path_matches))

    # 数据文件名
    curr_dir = os.getcwd()
    filename = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_FMA.csv')

    # 表头
    header = ['od', 'tf', 'let', 'prob', 'path']

    # 创建并写入CSV文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(header)

        # 写入数据
        for i in range(min_length):
            od = [int(x) for x in OD_matches[i].split(',')]
            let = float(LET_matches[i])
            tf = float(tf_matches[i])
            prob = float(prob_matches[i])
            path = [int(x) for x in path_matches[i].split(',')]

            writer.writerow([od, tf, let, prob, path])

    print(f"CSV文件 '{filename}' 已创建并写入数据.")

def handle_MLR_data(net_name, txt_name):
    # 读取文件内容
    with open(txt_name, 'r') as file:
        data = file.read()

    # 正则表达式模式
    OD_pattern = re.compile(r'OD=\[([0-9,\s]+)\]')
    LET_pattern = re.compile(r'LET=([\d.]+)')
    tf_pattern = re.compile(r'tf=([\d.]+)')
    prob_pattern = re.compile(r'MLR prob, g, t, path: \[\(([\d.]+)')
    # path_pattern = re.compile(r'DOT prob, g, t, path: \[[\d.,\s]+, \[([0-9,\s]+)\]\]')
    path_pattern = re.compile(r'MLR prob, g, t, path: \[\([\d.,\s]+\), \[([\d,\s]+)\]\]')


    # 匹配数据
    OD_matches = OD_pattern.findall(data)
    LET_matches = LET_pattern.findall(data)
    tf_matches = tf_pattern.findall(data)
    prob_matches = prob_pattern.findall(data)
    path_matches = path_pattern.findall(data)

    # 使用列表推导式将每个OD元素重复三次
    OD_matches = [od for od in OD_matches for _ in range(3)]

    # 确保所有匹配的列表长度相同
    min_length = min(len(OD_matches), len(LET_matches), len(tf_matches), len(prob_matches), len(path_matches))

    # 数据文件名
    curr_dir = os.getcwd()
    filename = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_OS-MIP.csv')

    # 表头
    header = ['od', 'tf', 'let', 'prob', 'path']

    # 创建并写入CSV文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(header)

        # 写入数据
        for i in range(min_length):
            od = [int(x) for x in OD_matches[i].split(',')]
            let = float(LET_matches[i])
            tf = float(tf_matches[i])
            prob = float(prob_matches[i])
            path = [int(x) for x in path_matches[i].split(',')]

            writer.writerow([od, tf, let, prob, path])

    print(f"CSV文件 '{filename}' 已创建并写入数据.")

def handle_PQL_data(net_name, txt_name):
    # 读取文件内容
    with open(txt_name, 'r') as file:
        data = file.read()

    # 正则表达式模式
    OD_pattern = re.compile(r'OD=\[([0-9,\s]+)\]')
    LET_pattern = re.compile(r'LET=([\d.]+)')
    tf_pattern = re.compile(r'tf=([\d.]+)')
    prob_pattern = re.compile(r'PQL prob, g, t, path: \(([\d.]+)')
    path_pattern = re.compile(r'PQL prob, g, t, path: \([\d.,\s]+, \[([\d,\s]+)\]\)')


    # 匹配数据
    OD_matches = OD_pattern.findall(data)
    LET_matches = LET_pattern.findall(data)
    tf_matches = tf_pattern.findall(data)
    prob_matches = prob_pattern.findall(data)
    path_matches = path_pattern.findall(data)

    # 确保所有匹配的列表长度相同
    min_length = min(len(OD_matches), len(LET_matches), len(tf_matches), len(prob_matches), len(path_matches))

    # 数据文件名
    curr_dir = os.getcwd()
    filename = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\csv\\{net_name}_PQL.csv')

    # 表头
    header = ['od', 'tf', 'let', 'prob', 'path']

    # 创建并写入CSV文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(header)

        # 写入数据
        for i in range(min_length):
            od = [int(x) for x in OD_matches[i].split(',')]
            let = float(LET_matches[i])
            tf = float(tf_matches[i])
            prob = float(prob_matches[i])
            path = [int(x) for x in path_matches[i].split(',')]

            writer.writerow([od, tf, let, prob, path])

    print(f"CSV文件 '{filename}' 已创建并写入数据.")


if __name__ == '__main__':
    curr_dir = os.getcwd()

    # ------------------------------------------DOT\FMA处理,使用时注释掉其他算法处理部分代码------------------------------------------------

    # net_name = 'SiouxFalls'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\S6_13_23_bm2.txt')

    # net_name = 'Friedrichshain'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\DOT&FMA.txt')

    # net_name = 'Anaheim'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\6_15_20_FMA_bm2.txt')

    # net_name = 'Winnipeg'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\6_15_16_bm2.txt')

    # handle_dot_data(net_name, txt_name)
    # handle_fma_data(net_name, txt_name)

    # ------------------------------------------OS-MIP处理,使用时注释掉其他算法处理部分代码------------------------------------------------
    # net_name = 'SiouxFalls'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\OS-MIP_6_14_15_bm2.txt')

    # net_name = 'Friedrichshain'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\OS-MIP.txt')

    # net_name = 'Anaheim'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\OS-MIP.txt')

    net_name = 'Winnipeg'
    txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\OS-MIP.txt')

    handle_MLR_data(net_name, txt_name)

    # ------------------------------------------PQL处理,使用时注释掉其他算法处理部分代码------------------------------------------------
    # net_name = 'SiouxFalls'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\S6_13_23_bm1.txt')

    # net_name = 'Friedrichshain'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\F6_13_23_bm1.txt')

    # net_name = 'Anaheim'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\6_14_14_bm1.txt')

    # net_name = 'Winnipeg'
    # txt_name = os.path.join(os.path.dirname(curr_dir), f'Networks\\{net_name}\\Benchmark_Record\\6_14_14_bm1.txt')
    #
    # handle_PQL_data(net_name, txt_name)