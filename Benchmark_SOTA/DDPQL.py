import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from utils.networks import Policy
from utils.replay_buffer import TRANSITION, PrioritizedEpisodeReplayMemory, EpisodeBuffer
from utils.cutom_env import *
from utils.misc import soft_update


class DDPQL(nn.Module):
    def __init__(self, n_node, time_budget, learning_rate=1e-2, device='cpu'):
        super(DDPQL, self).__init__()
        self.time_budget = time_budget
        self.device = device
        self.n_node = n_node
        self.model = Policy(state_size=n_node * 2 + 1, layer_size=256, action_size=n_node).to(device)
        self.target_model = Policy(state_size=n_node * 2 + 1, layer_size=256, action_size=n_node).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state, mask, exp=0.1):
        if random.random() > exp:
            logits = self.model(state) + (mask - 1) * 1e5
            probs = torch.softmax(torch.sigmoid(logits), dim=-1)
            action = torch.argmax(probs)
        else:
            candi_actions = mask.nonzero()
            action = random.choice(candi_actions)
        return action + 1

    def learn(self, episodes):
        bp_loss = []
        for episode, travel_time in episodes:
            seq_len = len(episode)
            batch = episode.sample()
            state_tensor = torch.FloatTensor(np.array(batch.state)).view(seq_len, -1).to(self.device)
            action_tensor = torch.LongTensor(np.array(batch.action) - 1).view(seq_len, -1).to(self.device)
            mask_tensor = torch.FloatTensor(np.array(batch.mask) - 1).view(seq_len, -1).to(self.device)
            next_mask_tensor = torch.FloatTensor(np.array(batch.prob) - 1).view(seq_len, -1).to(self.device)
            next_state_tensor = torch.FloatTensor(np.array(batch.next_state)).view(seq_len, -1).to(self.device)
            done = list(batch.done)

            state_val = self.model(state_tensor) + (mask_tensor - 1) * 1e5
            state_probs = torch.softmax(torch.sigmoid(state_val), dim=-1)
            state_action_prob = state_probs.gather(1, action_tensor)
            next_state_val = self.target_model(next_state_tensor)
            logits = next_state_val + (next_mask_tensor - 1) * 1e5
            next_probs = torch.softmax(torch.sigmoid(logits), dim=-1)
            next_action_tensor = torch.argmax(next_probs, dim=-1).view(seq_len, -1)
            next_state_action_prob = next_probs.gather(1, next_action_tensor)

            if done[-1] == 1 and travel_time < self.time_budget:
                next_state_action_prob[-1] = 1

            bp_loss.append(F.mse_loss(state_action_prob, next_state_action_prob.detach()).sum().view(1, -1))

        bp_loss = torch.concat(bp_loss)
        prios = bp_loss + 1e-8
        bp_loss = bp_loss.mean()
        self.optimizer.zero_grad()
        bp_loss.backward()
        self.optimizer.step()
        soft_update(self.target_model, self.model, 0.1)

        return bp_loss.item(), prios.data.cpu().numpy()


def main():
    map1 = MapInfo("maps/sioux_network.csv")
    env = Env(map1, 1, 15)
    env.reset()

    ddpql = DDPQL(n_node=env.map_info.n_node, time_budget=100, learning_rate=0.01, device='cpu')
    buffer = PrioritizedEpisodeReplayMemory(10000)

    for i_episode in range(1000):
        score = 0
        for _ in range(100):
            epi_buf = EpisodeBuffer()
            env.reset()
            state = env.get_agent_obs_onehot() + [ddpql.time_budget - env.cost_time]
            while True:
                state_tensor = torch.FloatTensor(state).to(ddpql.device)
                mask = env.get_agent_mask()
                mask_tensor = torch.FloatTensor(mask).to(ddpql.device)
                action = ddpql.select_action(state_tensor.unsqueeze(0), mask_tensor, exp=(1000-i_episode)/1000)
                _, cost, done = env.step(action.item())
                next_state = env.get_agent_obs_onehot() + [ddpql.time_budget - env.cost_time]
                next_mask = env.get_agent_mask()
                epi_buf.push(state, action, next_state, env.cost_time, mask, next_mask, done)
                state = next_state
                if done or len(env.path) > env.map_info.n_node:
                    score += 1 if env.cost_time < ddpql.time_budget else 0
                    break
            buffer.push(epi_buf, env.cost_time)

        samples, indices, _ = buffer.sample(100)
        loss, prios = ddpql.learn(samples)
        buffer.update_priorities(indices, prios)
        print('episodes:{}\tloss:{:.4f}\tscore: {:.4f}'.format(i_episode + 1, loss, score / 1000), env.path)


if __name__ == '__main__':
    main()
