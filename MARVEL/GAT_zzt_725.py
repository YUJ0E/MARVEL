import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import matplotlib.pyplot as plt
from torch import nn
import wandb
import time

class GAT(torch.nn.Module):
    def __init__(self, data_name, learning_rate, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6, alpha=1.0):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat_conv = GATConv(in_channels, int(in_channels/heads), heads=heads, dropout=dropout)
        self.gat_conv1 = GATConv(in_channels, int(in_channels/heads), heads=heads, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(in_channels)
        self.norm2 = torch.nn.LayerNorm(in_channels)
        self.gcn3 = GCNConv(in_channels, 2 * in_channels)
        self.gcn4 = GCNConv(2 * in_channels, 4 * in_channels)

        # 定义全连接层
        self.fc = nn.Linear(14, 14)
        wandb.init(project='MACTP_zzt_729', name=time.strftime('%m%d%H%M%S'+'_4layers'))
        wandb.config.update({
            "data_name": data_name,
            "learning_rate": learning_rate,
            "In Channels": in_channels
        })

    def forward(self, x, edge_index, current_node_indices):
        print('edge_index:', edge_index)
        # if current_node_indices == 2:
        #     optimal = x.gather(0, current_node_indices.view(-1, 1).expand(-1, x.size(1)))
        #     print('optimal:', optimal)
        residual = x
        x = self.gat_conv(x, edge_index)
        # x = F.relu(x)
        x = x + residual  # Residual connection
        x = self.norm1(x)  # Normalization
        # residual = x
        # x = self.gat_conv1(x, edge_index)
        # x = x + residual  # Residual connection
        #
        # x = self.norm2(x)  # Normalization

        x = self.gcn3(x, edge_index)

        x = F.relu(x)

        x = self.gcn4(x, edge_index)

        # 对输出 x 乘以其转置矩阵
        x = x@x.t()

        # print('x:', x)
        # 使用 gather 函数抽出对应当前位置的一行
        optimal = x.gather(0, current_node_indices.view(-1, 1).expand(-1, x.size(1)))

        # 通过全连接层生成 n*1 的张量
        # out = self.fc(current_node_embeddings)
        out = optimal.view(-1)
        return out

    def train_model(self, env, optimizer, criterion, num_episodes, device, scheduler, model_path, emb_size):
        success_rates = []
        total_losses = []
        success_count = 0
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            total_loss = 0
            done = False
            while not done:
                reward, done, loss, success = env.step(self, optimizer, criterion, device, model_path, emb_size)
                total_reward += reward
                total_loss += loss
                if success:
                    success_count += 1
            success_rate = success_count / (episode + 1)
            success_rates.append(success_rate)
            total_losses.append(total_loss)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Success Rate: {success_rate}, Total Loss: {total_loss}")
            scheduler.step()  # 每一轮结束后更新学习率

            wandb.log({
                "episode": episode + 1,
                "success_rate": success_rate,
                "total_loss": total_loss
            })

        self.plot_success_rates(success_rates)
        wandb.finish()

    def plot_success_rates(self, success_rates):
        plt.plot(success_rates)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Success Rate per Episode')
        plt.show()
