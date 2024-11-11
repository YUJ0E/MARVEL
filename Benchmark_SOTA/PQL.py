from CTD import Trainer
from utils.cutom_env import *


# class PQL:
#     def __init__(self, mymap, time_budget):
#         n_node = mymap.n_node
#         self.map_info = mymap
#         self.time_budget = time_budget
#         self.n_node = n_node
#         self.prob_table = [[random.random()] * n_node for _ in range(n_node)]
#
#     def select_action(self, location, travel_time, exp=0.):
#         next_nodes = self.map_info.get_next_nodes(location)
#         if random.random() > exp:
#             probs = np.array([self.prob_table[location - 1][n - 1] for n in next_nodes])
#             idx = np.argmax(probs)
#             action = next_nodes[idx]
#         else:
#             action = random.choice(next_nodes)
#         return int(action)
#
#     def update(self, state, action, travel_time, done, lr=0.1):
#         next_state = action
#         if not done:
#             next_action = self.select_action(next_state, travel_time, exp=0)
#             next_state_prob = self.prob_table[next_state - 1][next_action - 1]
#             self.prob_table[state - 1][action - 1] = lr * next_state_prob + (1 - lr) * self.prob_table[state - 1][action - 1]
#         elif travel_time < self.time_budget:
#             self.prob_table[state - 1][action - 1] = lr * 1 + (1 - lr) * self.prob_table[state - 1][action - 1]


class QL:
    def __init__(self, mymap, time_budget):
        n_node = mymap.n_node
        self.map_info = mymap
        self.time_budget = time_budget
        self.n_node = n_node
        self.q_table = [[random.random()] * n_node for _ in range(n_node)]

    def select_action(self, location, travel_time, exp=0.):
        next_nodes = self.map_info.get_next_nodes(location)
        if random.random() > exp:
            q_vals = np.array([self.q_table[location - 1][n - 1] for n in next_nodes])
            idx = np.argmin(q_vals)
            action = next_nodes[idx]
        else:
            action = random.choice(next_nodes)
        return int(action)

    def update(self, state, action, travel_time, done, lr=0.1, **kwargs):
        next_state = action
        reward = kwargs['cost']
        if not done:
            next_action = self.select_action(next_state, travel_time)
            next_state_prob = self.q_table[next_state - 1][next_action - 1]
            self.q_table[state - 1][action - 1] = lr * (reward + next_state_prob) + (1 - lr) * self.q_table[state - 1][action - 1]
        else:
            self.q_table[state - 1][action - 1] = lr * reward + (1 - lr) * self.q_table[state - 1][action - 1]


if __name__ == '__main__':
    map1 = MapInfo("D:/SE-GAC-master/nn/Chicago_Sketch/Chicago_Sketch_network.csv")
    env1 = Env(map1, 52, 289)
    T = env1.LET_cost[0] * 1.025
    # T = 0.95 * map1.get_let_time(52, 289)
    pql = Trainer(env1, policy=QL, time_budget=T, min_eps=0.1)
    pql.warm_start(1000, lr=0.5)
    pi_score = pql.train(num_train=1000, with_eval=False, int_eval=100)
    pql.eval(1000)
