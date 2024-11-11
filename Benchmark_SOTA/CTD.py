import random

from utils.cutom_env import *
from scipy.stats import norm
from tqdm import tqdm


class LinearDecay:
    """ Linearly Decays epsilon for exploration between a range of episodes"""

    def __init__(self, min_eps, max_eps, total_episodes):
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.total_episodes = total_episodes
        self.curr_episodes = 0
        # Todo: make 0.8 available as parameter
        self._threshold_episodes = 0.8 * total_episodes
        self.eps = max_eps

    def update(self):
        self.curr_episodes += 1
        eps = self.max_eps * (self._threshold_episodes - self.curr_episodes) / self._threshold_episodes
        self.eps = max(self.min_eps, eps)


class CTD:
    def __init__(self, mymap, time_budget):
        n_node = mymap.n_node
        self.map_info = mymap
        self.time_budget = time_budget
        self.n_node = n_node
        self.mu_table = [[random.random() * 1e7] * n_node for _ in range(n_node)]
        self.sigma2_table = [[random.random()] * n_node for _ in range(n_node)]

    def select_action(self, location, travel_time):
        next_nodes = self.map_info.get_next_nodes(location)
        # if random.random() > exp:
        # mus = np.array([self.map_info.G.get_edge_data(location-1, n-1)[0]["mu"] for n in next_nodes])
        mus = np.array([self.mu_table[location - 1][n - 1] for n in next_nodes])
        sigmas = np.sqrt([self.sigma2_table[location - 1][n - 1] for n in next_nodes])
        probs = norm(mus, sigmas).cdf(self.time_budget - travel_time)
        idx = np.argmax(probs)
        action = next_nodes[idx]
        # else:
        #     action = random.choice(next_nodes)
        return int(action)

    def update(self, state, action, done, lr=0.1, **kwargs):
        travel_time = kwargs['travel_time']
        next_state = action
        action_mu = self.map_info.G.get_edge_data(state - 1, next_state - 1)[0]["mu"]
        action_sigma2 = self.map_info.G.get_edge_data(state - 1, next_state - 1)[0]["sigma2"]
        if not done:
            state_mu = self.mu_table[state - 1][action - 1]
            state_sigma2 = self.sigma2_table[state - 1][action - 1]
            next_action = self.select_action(next_state, travel_time)
            next_state_mu = self.mu_table[next_state - 1][next_action - 1]
            next_state_sigma2 = self.sigma2_table[next_state - 1][next_action - 1]
            self.mu_table[state - 1][action - 1] = lr * (action_mu + next_state_mu) + (1 - lr) * \
                                                   self.mu_table[state - 1][action - 1]
            self.sigma2_table[state - 1][action - 1] = lr * (action_sigma2 + next_state_sigma2) + (1 - lr) * \
                                                       self.sigma2_table[state - 1][action - 1]
        elif travel_time < self.time_budget:
            self.mu_table[state - 1][action - 1] = lr * action_mu + (1 - lr) * self.mu_table[state - 1][action - 1]
            self.sigma2_table[state - 1][action - 1] = lr * action_sigma2 + (1 - lr) * self.sigma2_table[state - 1][
                action - 1]


class Trainer:
    def __init__(self, env, time_budget, policy, min_eps=0.1, max_eps=1):
        self.env = env
        self.env.reset()
        self.time_budget = time_budget
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.policy = policy(env.map_info, time_budget)

    def modify_time_budget(self, time_budget):
        self.time_budget = time_budget
        self.policy.time_budget = time_budget
        if self.with_critic:
            self.critic.time_budget = time_budget

    # def save(self, save_path):
    #     check_point = {'mu': self.policy.mu_table,
    #                    'sigma2': self.policy.sigma2_table}
    #     np.save(check_point, save_path)
    #
    # def load(self, load_path):
    #     check_point = np.load(load_path)
    #     self.policy.mu_table = check_point['mu']
    #     self.policy.sigma2_table = check_point['sigma2']

    def LET_step(self, exp=0):
        next_nodes = self.env.map_info.get_next_nodes(self.env.position)
        if random.random() > exp:
            action = self.env.LET_path[self.env.position - 1][1]
        else:
            action = random.choice(next_nodes)

        _, cost, done = self.env.step(action)
        next_state = self.env.position
        return action, next_state, cost, done



    def pql_step(self, state, exp=0):
        action = self.policy.select_action(state, self.env.cost_time)
        _, cost, done = self.env.step(action)
        next_state = self.env.position
        return action, next_state, cost, done


    def step(self, state, exp=0):
        next_nodes = self.env.map_info.get_next_nodes(self.env.position)
        if random.random() > exp:
            if random.random() > exp:
                action = self.policy.select_action(state, self.env.cost_time)
            else:
                action = random.choice(next_nodes)
        else:
            action = self.env.LET_path[self.env.position - 1][1]

        _, cost, done = self.env.step(action)
        next_state = self.env.position
        return action, next_state, cost, done
    
    # def pql_step(self, state, exp=0):
    #     action = self.policy.select_action(state, self.env.cost_time, exp)
    #     _, cost, done = self.env.step(action)
    #     next_state = self.env.position
    #     return action, next_state, cost, done

    def eval(self, num_eval):
        best_score = 0
        scores = []
        pi_score = []
        # print('-----------------Evaling-----------------')
        for i_episode in range(num_eval):
            self.env.reset()
            state = self.env.position
            while True:
                # action, next_state, cost, done = self.pql_step(state) ###PQL
                action, next_state, cost, done = self.step(state)  ###CTD
                state = next_state
                if done ==1 or self.env.cost_time > self.time_budget:
                    scores.append(1 if self.env.cost_time < self.policy.time_budget else 0)
                    break
                elif done == 2:
                    scores.append(0)
            # flag = self.env.path
            # print(flag)
            # t = self.env.cost_time
        score_mean = np.mean(scores)
        if score_mean >= best_score:
            best_score = score_mean
        pi_score.append(score_mean)
        print('Eval_Finished\tscore_mean:{:.4f}'.format(score_mean))
        return pi_score

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

    def train(self, num_train, with_eval=False, int_eval=10, lr=0.1):
        best_score = 0
        pi_score = []
        scores = []
        self.eps = LinearDecay(self.min_eps, self.max_eps, num_train)
        for i_episode in range(num_train):
            if with_eval and i_episode % int_eval == 0:
                self.eval(100)

            self.env.reset()
            state = self.env.position
            while True:
                action, next_state, cost, done = self.step(state, self.eps.eps) ####CTD
                # action, next_state, cost, done = self.pql_step(state, self.eps.eps)####PQL
                self.policy.update(state=state, action=action, done=done, lr=lr,
                                   next_state=next_state, cost=cost, travel_time=self.env.cost_time)
                state = next_state
                if done or self.env.cost_time > self.time_budget:
                    scores.append(1 if self.env.cost_time < self.policy.time_budget else 0)
                    break

            score_mean = np.mean(scores)
            if score_mean >= best_score:
                best_score = score_mean
            pi_score.append(score_mean)
            if i_episode % int_eval == 0:
                print('Train_Episodes:{}\tscore: {:.3f}\tbest score:{:.3f}\tlast path:{}'
                      .format(i_episode + 1, score_mean, best_score, self.env.path), end='\n')
            self.eps.update()
        return pi_score


if __name__ == '__main__':
    map1 = MapInfo("D:/SE-GAC-master/nn/Anaheim/Anaheim_network.csv")
    env1 = Env(map1, 96, 161)
    T = 1.05 * map1.get_let_time(96, 161)
    ctd = Trainer(env1, policy=CTD, time_budget=T, min_eps=0.1, max_eps=1)
    ctd.warm_start(1000, lr=0.5)
    pi_score = ctd.train(num_train=1000, with_eval=False, int_eval=1)
    ctd.eval(1000)
