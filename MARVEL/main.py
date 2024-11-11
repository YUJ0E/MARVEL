import torch
import torch.optim as optim
import torch.nn as nn
from Env_724 import Env
from GCN import GCN
from GAT_zzt_725 import GAT

def main():
    data_name = 'Simple'
    model_path = 'your_model_path'
    emb_size = 128
    robots = [{'robot_num': 1,'importance': 1, 'max_time':4.8, 'time_spent': 0, 'start_node': 9, 'goal_node': 12, 'current_node': 9, 'next_node': None},
              {'robot_num': 2,'importance': 1, 'max_time':8.9, 'time_spent': 0,'start_node': 1, 'goal_node': 8, 'current_node': 1, 'next_node': None}]
    for robot in robots:
        robot['time_spent'] = torch.tensor(robot['time_spent'], dtype=torch.float)

    env = Env(data_name, model_path, emb_size, robots, device)

    in_channels = emb_size + 40
    hidden_channels = 2 * in_channels
    out_channels = 1
    learning_rate = 1e-5
    # model = GCN(in_channels=emb_size + 128, hidden_channels=256, out_channels=1).to(device)  # 使用 GCN 模型
    model = GAT(data_name, learning_rate, in_channels, hidden_channels, out_channels, heads=6).to(device)  # 使用 GAT 模型
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)  # 每10轮将学习率减少到原来的95%
    criterion = nn.CrossEntropyLoss()

    model.train_model(env, optimizer, criterion, num_episodes=100000, device=device, scheduler=scheduler, model_path=model_path, emb_size=emb_size)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
