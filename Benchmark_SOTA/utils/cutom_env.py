import copy
import numpy as np
import networkx as nx
import random
from utils.func import MapInfo


class Env:
    def __init__(self, map_info: MapInfo, origin, destination, max_steps=50):
        self.map_info = map_info
        self.origin = origin
        self.destination = destination
        self.position = origin
        self.LET_path = [None] * self.map_info.n_node
        self.LET_cost = [0] * self.map_info.n_node
        self._get_letpaths()
        self.step_cnt = 0
        self.done = 0
        self.max_steps = max_steps
        self.cost_time = 0
        self.path = [self.position]
        self.reset()

    def _get_letpaths(self):
        for i in range(self.map_info.n_node):
            if i in nx.nodes(self.map_info.G) and nx.has_path(self.map_info.G, i, self.destination-1):
                self.LET_path[i] = self.map_info.get_let_path(i+1, self.destination)
                self.LET_cost[i] = self.map_info.get_let_time(i+1, self.destination)

    def __update_agent_pos(self, n_pos):
        parti_tmp = self.map_info.get_next_nodes(self.position)
        if n_pos not in parti_tmp:
            raise Exception('Edge[{}, {}] Not Found!'.format(self.position), n_pos)
        self.position = n_pos

    def get_agent_mask(self, one_hot=True):
        mask = self.map_info.get_next_nodes(self.position, zero_mask=one_hot)
        return mask

    def get_agent_obs(self):
        obs = [self.position, self.destination]
        return obs

    def get_agent_obs_onehot(self):
        obs_onehot = list(np.eye(self.map_info.n_node)[self.position-1]) + \
              list(np.eye(self.map_info.n_node)[self.destination-1])
        return obs_onehot

    # action should be a edge
    def step(self, action):
        cost = self.map_info.get_edge_cost([self.position, action])
        self.__update_agent_pos(action)

        obs = self.get_agent_obs()
        if self.step_cnt >= self.max_steps:
            self.done = 2
        elif self.position == self.destination:
            self.done = 1

        self.step_cnt += 1
        self.cost_time += cost
        self.path.append(self.position)

        return obs, cost, self.done

    def reset(self):
        self.step_cnt = 0
        self.done = 0
        self.cost_time = 0
        self.position = self.origin
        self.path = [self.position]
        return self.get_agent_obs()

    def all_reset(self, origin, destination):
        self.origin = origin
        self.destination = destination
        self.LET_path = [None] * self.map_info.n_node
        self.LET_cost = [0] * self.map_info.n_node
        self._get_letpaths()
        self.reset()

    def render(self):
        print("\r step_count:{}; agents_pos:{};".format(self.step_cnt, self.position))

    def close(self):
        del self

# map1 = MapInfo()
# env = Env(map1, 1, 15)
# env.reset()
