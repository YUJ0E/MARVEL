import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils.misc import CategoricalMasked
from utils.networks import Policy
from utils.replay_buffer import TRANSITION, EpisodeReplayMemory, EpisodeBuffer
from utils.cutom_env import *

WITH_WIS = False


class GeneralizedPG(nn.Module):
    def __init__(self, n_node, time_budget, learning_rate=1e-2, device='cpu'):
        super(GeneralizedPG, self).__init__()
        self.time_budget = time_budget
        self.device = device
        self.n_node = n_node
        self.policy = Policy(state_size=n_node * 2 + 1, layer_size=128, action_size=n_node).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, mask):
        logits = self.policy(state) + (mask - 1) * 1e5
        m = Categorical(logits=logits)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item() + 1, log_prob

    def get_log_prob(self, state, mask, action):
        logits = self.policy(state) + (mask - 1) * 1e5
        m = Categorical(logits=logits)
        log_prob = m.log_prob(action.view(-1) - 1)
        return log_prob

    def learn(self, episodes, critic=None, now_prob=False, **kwargs):
        policy_loss = []
        for episode, travel_time in episodes:
            bp = 0.1
            seq_len = len(episode)
            batch = episode.sample()
            if now_prob:
                state_tensor = torch.FloatTensor(np.array(batch.state)).view(seq_len, -1).to(self.device)
                actoin_tensor = torch.FloatTensor(np.array(batch.action)).view(seq_len, -1).to(self.device)
                mask_tensor = torch.FloatTensor(np.array(batch.mask)).view(seq_len, -1).to(self.device)
                log_prob_tensor = self.get_log_prob(state_tensor.detach(), mask_tensor.detach(),
                                                    actoin_tensor.detach()).view(seq_len, -1)
            else:
                log_prob_tensor = torch.clone(torch.cat(batch.prob).view(seq_len, -1)).to(self.device)
            o_state_tensor = torch.FloatTensor(np.array(batch.state[0])).view(1, -1).to(self.device)
            if critic:
                bp = critic.model(o_state_tensor)
                bp = 0.1 if bp < 0.1 else bp

            err_time = torch.FloatTensor([self.time_budget - travel_time]).to(self.device)
            R = (err_time > 0) * 1 - bp
            # R = (torch.sigmoid(err_time) - bp) if err_time < 0 else torch.FloatTensor([1]).to(self.device)
            policy_loss.append((-log_prob_tensor * R.detach()).sum().view(1, -1))

        self.optimizer.zero_grad()
        policy_loss = torch.concat(policy_loss).mean()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()


class OffGeneralizedPG(GeneralizedPG):
    def get_cur_log_probs(self, episodes):
        cur_log_probs = []
        for episode, travel_timein in episodes:
            seq_len = len(episode)
            batch = episode.sample()

            state_tensor = torch.FloatTensor(np.array(batch.state)).view(seq_len, -1).to(self.device)
            actoin_tensor = torch.FloatTensor(np.array(batch.action)).view(seq_len, -1).to(self.device)
            mask_tensor = torch.FloatTensor(np.array(batch.mask)).view(seq_len, -1).to(self.device)

            cur_log_prob = self.get_log_prob(state_tensor, mask_tensor, actoin_tensor).view(seq_len, -1)
            cur_log_probs.append(cur_log_prob)
        return cur_log_probs

    def get_cur_raus(self, episodes):
        raus = []
        for episode, travel_timein in episodes:
            seq_len = len(episode)
            batch = episode.sample()

            state_tensor = torch.FloatTensor(np.array(batch.state)).view(seq_len, -1).to(self.device)
            actoin_tensor = torch.FloatTensor(np.array(batch.action)).view(seq_len, -1).to(self.device)
            mask_tensor = torch.FloatTensor(np.array(batch.mask)).view(seq_len, -1).to(self.device)

            cur_log_prob = self.get_log_prob(state_tensor, mask_tensor, actoin_tensor).view(seq_len, -1)
            log_prob_tensor = torch.clone(torch.cat(batch.prob).view(seq_len, -1)).to(self.device)
            rau = torch.exp(cur_log_prob) / torch.exp(log_prob_tensor)
            raus.append(rau)

        return raus

    def learn(self, episodes, critic=None, **kwargs):
        policy_loss = []
        for (episode, travel_time), cur_log_prob in zip(episodes, kwargs['cur_log_probs']):
            bp = 0.1
            seq_len = len(episode)
            batch = episode.sample()

            o_state_tensor = torch.FloatTensor(np.array(batch.state[0])).view(1, -1).to(self.device)
            log_prob_tensor = torch.clone(torch.cat(batch.prob).view(seq_len, -1)).to(self.device)
            rau = torch.exp(cur_log_prob) / torch.exp(log_prob_tensor)
            rau = torch.cumprod(rau, dim=0)
            if WITH_WIS:
                rau = rau / rau.mean()

            if critic:
                bp = critic.model(o_state_tensor)
                bp = 0.1 if bp < 0.1 else bp

            err_time = torch.FloatTensor([self.time_budget - travel_time]).to(self.device)
            R = (err_time > 0) * 1 - bp
            # R = (torch.sigmoid(err_time) - bp) if err_time < 0 else torch.FloatTensor([1]).to(self.device)
            policy_loss.append((-cur_log_prob * R.detach() * rau.detach()).sum().view(1, -1))

        self.optimizer.zero_grad()
        policy_loss = torch.concat(policy_loss).mean()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

#
# def main():
#     map1 = MapInfo("maps/sioux_network.csv")
#     env = Env(map1, 1, 15)
#     env.reset()
#
#     gpg = OffGeneralizedPG(n_node=env.map_info.n_node, time_budget=40, learning_rate=0.01, device='cpu')
#     buffer = EpisodeReplayMemory(10000)
#
#     for i_episode in range(100):
#         score = 0
#         for _ in range(1000):
#             epi_buf = EpisodeBuffer()
#             env.reset()
#             state = env.get_agent_obs_onehot() + [gpg.time_budget - env.cost_time]
#             while True:
#                 state_tensor = torch.FloatTensor(state).to(gpg.device)
#                 mask = env.get_agent_mask()
#                 mask_tensor = torch.FloatTensor(mask).to(gpg.device)
#                 action, log_prob = gpg.select_action(state_tensor.unsqueeze(0), mask_tensor)
#                 _, cost, done = env.step(action)
#                 next_state = env.get_agent_obs_onehot() + [gpg.time_budget - env.cost_time]
#                 next_mask = env.get_agent_mask()
#                 epi_buf.push(state, action, next_state, env.cost_time, mask, log_prob, done)
#                 state = next_state
#                 if done or len(env.path) > env.map_info.n_node:
#                     score += 1 if env.cost_time < gpg.time_budget else 0
#                     break
#             buffer.push(epi_buf, env.cost_time)
#         print('episodes:{}\tscore: {}'.format(i_episode + 1, score / 1000), env.path)
#         samples = buffer.sample(1000)
#         # cur_log_probs = None
#         cur_log_probs = gpg.get_cur_log_probs(samples)
#         gpg.learn(samples, cur_log_probs=cur_log_probs)
#         # buffer.clear()
#
#
# if __name__ == '__main__':
#     main()
