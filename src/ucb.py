from collections import deque
import numpy as np

class UCB():
    def __init__(self, n, size = 90, epsilon = 0.5, beta = 1.):
        self.rewards = deque(maxlen = size)
        self.idxs = deque(maxlen = size)
        self.n = n
        self.size = size
        self.epsilon = epsilon
        self.beta = beta
        self.count = 0

    def get_idx(self):
        if self.count < self.n:
            self.count += 1
            return self.count - 1
        if np.random.rand() > self.epsilon:
            N = np.zeros((self.n, ))
            rewards = np.zeros((self.n, ))
            for i, reward in zip(self.idxs, self.rewards):
                N[i] += 1
                rewards[i] += reward
            rewards = rewards / (N + 1e-9)

            values = rewards + self.beta * np.sqrt(1 / (N + 1e-9))
            best = np.flatnonzero(values == values.max())
            return np.random.choice(best)
        else:
            return np.random.choice(self.n)

    def push(self, idx, reward):
        self.idxs.append(idx)
        self.rewards.append(reward)