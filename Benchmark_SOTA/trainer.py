import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GPG import GeneralizedPG, OffGeneralizedPG
from ActorCritic import ActorCritic
from EVFA import ExtendedVFA, OffExtendedVFA
from utils.replay_buffer import EpisodeReplayMemory, EpisodeBuffer, PrioritizedEpisodeReplayMemory
from utils.cutom_env import *
from utils import Logger
from tqdm import tqdm
import networkx as nx
import os


class GeneralizedAC:
    def __init__(self, env, time_budget, buffer_size=100, lr=1e-2, mode='on-policy', with_critic=True, device='cpu', ckpt=None):
        self.env = env
        self.env.reset()
        self.mode = mode
        self.with_critic = with_critic
        self.buffer_size = buffer_size
        if mode == 'on-policy':
            self.policy = GeneralizedPG(n_node=env.map_info.n_node, time_budget=time_budget,
                                        learning_rate=lr, device=device)
            self.critic = ExtendedVFA(n_node=env.map_info.n_node, time_budget=time_budget,
                                      learning_rate=lr, device=device) \
                if with_critic else None
            self.buffer = EpisodeReplayMemory(buffer_size)
        elif mode == 'off-policy':
            self.policy = OffGeneralizedPG(n_node=env.map_info.n_node, time_budget=time_budget,
                                           learning_rate=lr, device=device)
            self.critic = OffExtendedVFA(n_node=env.map_info.n_node, time_budget=time_budget,
                                         learning_rate=lr, device=device) \
                if with_critic else None
            self.buffer = PrioritizedEpisodeReplayMemory(buffer_size)
        elif mode == 'let-ac':
            self.policy = GeneralizedPG(n_node=env.map_info.n_node, time_budget=time_budget,
                                        learning_rate=lr, device=device)
            self.critic = None
            self.buffer = EpisodeReplayMemory(buffer_size)
        if ckpt:
            self.load(ckpt)
        self.time_budget = time_budget
        self.device = device

    def modify_time_budget(self, time_budget):
        self.time_budget = time_budget
        self.policy.time_budget = time_budget
        if self.with_critic:
            self.critic.time_budget = time_budget

    def save(self, save_path):
        check_point = {'policy': self.policy.state_dict(),
                       'critic': self.critic.state_dict() if self.with_critic else None}
        torch.save(check_point, save_path)

    def load(self, load_path):
        check_point = torch.load(load_path)
        self.policy.load_state_dict(check_point['policy'])
        if self.with_critic:
            self.critic.load_state_dict(check_point['critic'])

    def step(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask = self.env.get_agent_mask()
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        action, log_prob = self.policy.select_action(state_tensor.unsqueeze(0), mask_tensor)
        if action in self.env.path:
            return self.LET_step(state)
        _, cost, done = self.env.step(action)
        next_state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
        return action, next_state, mask, log_prob, done

    def LET_step(self, state, eps=0.):
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask = self.env.get_agent_mask()
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        if random.random() > eps:
            action = self.env.LET_path[self.env.position - 1][1]
        else:
            candi_actions = self.env.map_info.get_next_nodes(self.env.position)
            action = random.choice(candi_actions)
        action_tensor = torch.FloatTensor([action]).to(self.device)
        log_prob = self.policy.get_log_prob(state_tensor, mask_tensor, action_tensor)
        _, cost, done = self.env.step(action)
        next_state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
        return action, next_state, mask, log_prob, done

    def eval(self, num_eval, epi_size):
        best_score = 0
        scores = []
        # print('-----------------Evaling-----------------')
        for i_episode in range(num_eval):
            score = []
            for _ in range(epi_size):
                self.env.reset()
                state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
                while True:
                    action, next_state, mask, log_prob, done = self.step(state)
                    state = next_state
                    if done or self.env.cost_time > self.time_budget:
                        score.append(1 if self.env.cost_time < self.policy.time_budget else 0)
                        break
            score = np.mean(score)
            if score >= best_score:
                best_score = score
            scores.append(score)
            # print('Eval_Episodes:{}\tscore: {:.3f}\tbest score:{:.3f}\tlast path:{}'.format(
            #     i_episode + 1, score, best_score, self.env.path))
        print('Eval_Finished\tscore_mean:{:.4f}\tscore_std:{:.4f}'.format(np.mean(scores), np.std(scores)))
        # print('-----------------Evaling-----------------')
        return scores

    def warm_start(self, num_train=100, batch_size=100, epsilon=0, start_node=None, destination_node=None, save=False):
        cur_dir = os.getcwd()
        dir_path = cur_dir + '\OD=' + str(start_node) + '-' + str(destination_node)
        os.mkdir(dir_path)
        print('*********************warm-start is running***************************')
        for _ in range(self.buffer_size):
            epi_buf = EpisodeBuffer()
            self.env.reset()
            state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
            while True:
                action, next_state, mask, log_prob, done = self.LET_step(state, eps=epsilon)
                if self.mode == 'let-ac':
                    let_cost = self.env.LET_cost[self.env.position - 1]
                    epi_buf.push(state, action, next_state, self.env.cost_time, mask, log_prob, let_cost)
                else:
                    epi_buf.push(state, action, next_state, self.env.cost_time, mask, log_prob, done)
                state = next_state
                if done or len(self.env.path) > self.env.map_info.n_node:
                    break
            self.buffer.push(epi_buf, self.env.cost_time)
        for i_episode in tqdm(range(num_train)):
            save_path = os.path.join(dir_path, str(start_node) + '-' + str(destination_node) +'_warm' + '_episode=' + str(i_episode) + '.pth')
            if self.mode == 'on-policy' or self.mode == 'let-ac':
                samples = self.buffer.sample(batch_size)
                self.policy.learn(samples, now_prob=True)
            if self.mode == 'off-policy':
                samples, indices, _ = self.buffer.sample(batch_size)
                cur_log_probs = self.policy.get_cur_log_probs(samples)
                self.policy.learn(samples, critic=self.critic, cur_log_probs=cur_log_probs)
            if save:       
                self.save(save_path)
        print('*********************warm-start is finished**************************')

    def supervised_warm_start(self, num_train=100, destination_node=None, save=False):
        save_path = 'D=' + str(destination_node)+'_warm_params.pth'
        print('*********************warm-start is running***************************')
        data_list = []
        label_list = []
        n_node = self.env.map_info.n_node
        optimizer = optim.Adam(self.policy.policy.parameters(), lr=0.01)
        for i in range(self.env.map_info.n_node):
            self.env.position = i + 1
            if i + 1 == self.env.destination:
                continue
            state = torch.FloatTensor(self.env.get_agent_obs_onehot() + [self.env.LET_cost[i] * random.random()]).to(
                self.device)
            label = torch.LongTensor([self.env.LET_path[i][1] - 1]).to(self.device)
            data_list.append(state)
            label_list.append(label)
        data_tensor = torch.concat(data_list).view(n_node - 1, -1)
        label_tensor = torch.concat(label_list).view(-1)
        for _ in tqdm(range(num_train)):
            output_tensor = self.policy.policy(data_tensor.detach())
            output_tensor = F.softmax(F.sigmoid(output_tensor), dim=-1)
            loss = F.cross_entropy(output_tensor, label_tensor.detach())
            optimizer.zero_grad()
            # loss = torch.concat(loss).mean()
            loss.backward()
            optimizer.step()
            # print(loss.item())
        if save:       
            self.save(save_path)
        print('*********************warm-start is finished**************************')

    def train(self, num_train, batch_size=100, with_eval=False, int_eval=10, start_node=None, destination_node=None, save=False):
        best_score = 0
        pi_score = []
        cur_dir = os.getcwd()
        dir_path = cur_dir + '\OD=' + str(start_node) + '-' + str(destination_node)
        # os.mkdir(dir_path)
        for i_episode in tqdm(range(num_train)):
            save_path = os.path.join(dir_path, str(start_node) + '-' + str(destination_node) + '_episode=' + str(i_episode) + '.pth')
            score = []
            if with_eval and i_episode % int_eval == 0:
                self.eval(10, 100)
            for _ in range(batch_size):
                epi_buf = EpisodeBuffer()
                self.env.reset()
                state = self.env.get_agent_obs_onehot() + [self.time_budget - self.env.cost_time]
                while True:
                    action, next_state, mask, log_prob, done = self.step(state)
                    if self.mode == 'let-ac':
                        let_cost = self.env.LET_cost[self.env.position - 1]
                        epi_buf.push(state, action, next_state, self.env.cost_time, mask, log_prob, let_cost)
                    else:
                        epi_buf.push(state, action, next_state, self.env.cost_time, mask, log_prob, done)
                    state = next_state
                    if done or len(self.env.path) > self.env.map_info.n_node or self.env.cost_time > self.time_budget:
                        score.append(1 if self.env.cost_time < self.time_budget else 0)
                        break
                self.buffer.push(epi_buf, self.env.cost_time)

            if self.mode == 'off-policy':
                samples, indices, _ = self.buffer.sample(batch_size)
                cur_log_probs = self.policy.get_cur_log_probs(samples)
                c_loss = 0
                if self.critic:
                    c_loss, prios = self.critic.learn(samples, cur_log_probs=cur_log_probs)
                    self.buffer.update_priorities(indices, prios)
                p_loss = self.policy.learn(samples, critic=self.critic, cur_log_probs=cur_log_probs)

            elif self.mode == 'on-policy':
                samples = self.buffer.sample(self.buffer_size)
                c_loss = 0
                if self.critic:
                    c_loss = self.critic.learn(samples)
                p_loss = self.policy.learn(samples, critic=self.critic)

            elif self.mode == 'let-ac':
                samples = self.buffer.sample(self.buffer_size)
                c_loss = 0
                p_loss = self.policy.learn(samples)

            score_mean = np.mean(score)
            if score_mean >= best_score:
                best_score = score_mean

            if i_episode % int_eval == 0:
                print('Train_Episodes:{}\tscore: {:.3f}\t'
                      'best score:{:.3f}\tp_loss:{:.4f}\tc_loss:{:.4f}\tlast path:{}'
                      .format(i_episode + 1, score_mean,
                              best_score, p_loss, c_loss, self.env.path), end='\n')
            pi_score.append(score_mean)
            if save:       
                self.save(save_path)
        self.buffer.clear()
        return pi_score
