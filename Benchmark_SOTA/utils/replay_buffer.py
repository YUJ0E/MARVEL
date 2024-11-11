from collections import namedtuple
import random
import numpy as np


TRANSITION = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'travel_time', 'mask', 'prob', 'done'))


class EpisodeBuffer:
    def __init__(self):
        self.memory = []
        self.Transition = TRANSITION
        self.position = 0

    def push(self, state, action, next_state, cost, mask, prob, done):
        self.memory.append(None)
        self.memory[self.position] = self.Transition(state, action, next_state, cost, mask, prob, done)
        self.position = self.position + 1

    def sample(self):
        transitions = self.memory[:]
        batch = self.Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class EpisodeReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode, cost_sum):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode, cost_sum
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.memory, batch_size)
        return sampled_episodes

    def clear(self):
        self.position = 0
        del self.memory[:]

    def __len__(self):
        return len(self.memory)


class PrioritizedEpisodeReplayMemory:
    def __init__(self, capacity, prob_alpha=0.6):
        self.capacity = capacity
        self.prob_alpha = prob_alpha
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, episode, cost_sum):
        max_prio = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode, cost_sum
        self.position = (self.position + 1) % self.capacity

        self.priorities[self.position] = max_prio

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.prob_alpha
        probs /= np.sum(probs)
        indices = np.random.choice(len(self.memory), batch_size, p=probs if np.nansum(probs) == 1 else None)
        sampled_episodes = [self.memory[idx] for idx in indices]

        # sampled_episodes = random.sample(self.memory, batch_size)

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        return sampled_episodes, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def clear(self):
        self.position = 0
        del self.memory[:]

    def __len__(self):
        return len(self.memory)

