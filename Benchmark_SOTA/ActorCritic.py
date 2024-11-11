import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from GPG import GeneralizedPG
from utils.misc import CategoricalMasked
from utils.networks import Policy
from utils.replay_buffer import TRANSITION, EpisodeReplayMemory, EpisodeBuffer
from utils.cutom_env import *
import time
from tqdm import tqdm
WITH_WIS = False


class ActorCritic(GeneralizedPG):
    def __init__(self, n_node, time_budget, learning_rate=1e-2, device='cpu', env=None):
        super(ActorCritic, self).__init__(n_node, time_budget, learning_rate=1e-2, device='cpu')
        self.env = env

    def select_action(self, state, mask):
        return super().select_action(state, mask)

    def learn(self, episodes):
        policy_loss = []
        for episode, travel_time in episodes:
            seq_len = len(episode)
            batch = episode.sample()
            log_prob_tensor = torch.clone(torch.cat(batch.prob).view(seq_len, -1)).to(self.device)
            returns = np.ascontiguousarray(np.flip(np.array(batch.travel_time)))
            returns_tensor = torch.FloatTensor(returns).view(seq_len, -1).to(self.device)
            critic = np.ascontiguousarray(np.flip(np.array(batch.done)))
            critic_tensor = torch.FloatTensor(critic).view(seq_len, -1).to(self.device)

            policy_loss.append((-log_prob_tensor * (critic_tensor - returns_tensor).detach()).sum().view(1, -1))

        self.optimizer.zero_grad()
        policy_loss = torch.concat(policy_loss).mean()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

    def LET_step(self, exp=0):
        next_nodes = self.env.map_info.get_next_nodes(self.env.position)
        if random.random() > exp:
            action = self.env.LET_path[self.env.position - 1][1]
        else:
            action = random.choice(next_nodes)

        _, cost, done = self.env.step(action)
        next_state = self.env.position
        return action, next_state, cost, done


    def warm_start(self, num_train, lr=0.1):
        # self.eps = LinearDecay(self.min_eps, self.max_eps, num_train)
        pi_score = []
        scores = []
        for i_episode in tqdm(range(num_train)):
            self.env.reset()
            state = self.env.position
            while True:
                action, next_state, cost, done = self.LET_step(0.1)
                self.policy.update(state=state, action=action, done=done, lr=lr,
                                   next_state=next_state, cost=cost, travel_time=self.env.cost_time)
                state = next_state
                if done or self.env.cost_time > self.time_budget:
                    scores.append(1 if self.env.cost_time < self.policy.time_budget else 0)
                    break
            score_mean = np.mean(scores)
            pi_score.append(score_mean)
            if i_episode % (num_train // 100) == 0:
                print('Train_Episodes:{}\tscore: {:.3f}\tlast path:{}'
                      .format(i_episode + 1, score_mean, self.env.path), end='\n')
            # self.eps.update()    







    def train(self, with_eval=False):
        self.env.reset()
        buffer = EpisodeReplayMemory(100)
        for i_episode in range(0):
            score = 0
            for _ in range(100):
                epi_buf = EpisodeBuffer()
                self.env.reset()
                state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
                while True:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    mask = self.env.get_agent_mask()
                    mask_tensor = torch.FloatTensor(mask).to(self.device)
                    action, log_prob = self.select_action(state_tensor.unsqueeze(0), mask_tensor)
                    _, cost, done = self.env.step(action)
                    next_state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
                    LET_cost = self.env.LET_cost[self.env.position - 1]
                    epi_buf.push(state, action, next_state, self.env.cost_time, mask, log_prob, LET_cost)
                    state = next_state
                    if done or len(self.env.path) > self.env.map_info.n_node:
                        score += 1 if self.env.cost_time < self.time_budget else 0
                        break
                buffer.push(epi_buf, self.env.cost_time)
            print('episodes:{}\tscore: {}'.format(i_episode + 1, score / 100), self.env.path)
            samples = buffer.sample(100)
            self.learn(samples)

        if with_eval:
            t1 = time.perf_counter()
            score = 0
            for _ in range(1000):
                epi_buf = EpisodeBuffer()
                self.env.reset()
                state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
                while True:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    mask = self.env.get_agent_mask()
                    mask_tensor = torch.FloatTensor(mask).to(self.device)
                    action, log_prob = self.select_action(state_tensor.unsqueeze(0), mask_tensor)
                    _, cost, done = self.env.step(action)
                    next_state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
                    LET_cost = self.env.LET_cost[self.env.position - 1]
                    epi_buf.push(state, action, next_state, self.env.cost_time, mask, log_prob, LET_cost)
                    state = next_state
                    if done or len(self.env.path) > self.env.map_info.n_node:
                        score += 1 if self.env.cost_time < self.time_budget else 0
                        break
                buffer.push(epi_buf, self.env.cost_time)
            prob = score / 1000
            print("-----------------eval------------------")
            print('prob=: {}'.format(prob), self.env.path)
            t = time.perf_counter() - t1
            return prob, t





# def main():
#     map1 = MapInfo("maps/sioux_network.csv")
#     env1 = Env(map1, 1, 15)
#     env1.reset()

#     ac = ActorCritic(n_node=env1.map_info.n_node, time_budget=40, learning_rate=0.01, device='cpu', env=env1)
#     buffer = EpisodeReplayMemory(100)

#     for i_episode in range(100):
#         score = 0
#         for _ in range(100):
#             epi_buf = EpisodeBuffer()
#             env1.reset()
#             state = env1.get_agent_obs_onehot() + [ac.time_budget - env1.cost_time]
#             while True:
#                 state_tensor = torch.FloatTensor(state).to(ac.device)
#                 mask = env1.get_agent_mask()
#                 mask_tensor = torch.FloatTensor(mask).to(ac.device)
#                 action, log_prob = ac.select_action(state_tensor.unsqueeze(0), mask_tensor)
#                 _, cost, done = env1.step(action)
#                 next_state = env1.get_agent_obs_onehot() + [ac.time_budget - env1.cost_time]
#                 LET_cost = env1.LET_cost[env1.position - 1]
#                 epi_buf.push(state, action, next_state, env1.cost_time, mask, log_prob, LET_cost)
#                 state = next_state
#                 if done or len(env1.path) > env1.map_info.n_node:
#                     score += 1 if env1.cost_time < ac.time_budget else 0
#                     break
#             buffer.push(epi_buf, env1.cost_time)
#         print('episodes:{}\tscore: {}'.format(i_episode + 1, score / 1000), env1.path)
#         samples = buffer.sample(100)
#         ac.learn(samples)


if __name__ == '__main__':
    map1 = MapInfo("maps/sioux_network.csv")
    env1 = Env(map1, 1, 15)
    env1.reset()
    ac = ActorCritic(n_node=env1.map_info.n_node, time_budget=45, learning_rate=0.01, device='cpu', env=env1)
    prob, t = ac.train(with_eval=True)

    print(prob)
    print(t)

