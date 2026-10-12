import numpy as np

class Episode_memory():
    def __init__(self, num_envs, m, l, n):
        self.num_envs = num_envs
        self.m = m
        self.l = l
        self.n = n
        self.lens = [self.m for _ in range(self.num_envs)]
        self.reset()

    def save(self, states, actions, rs_e, rs_i, dones, values, target_values, values_int, target_values_int,
             h, c, h_int, c_int, last_actions, last_rs_e, last_rs_i):
        for i in range(self.num_envs):
            self.lens[i] += 1
            self.states[i].append(states[i])
            self.actions[i].append(actions[i])
            self.rs_e[i].append(rs_e[i])
            self.rs_i[i].append(rs_i[i])
            self.dones[i].append(dones[i])
            self.values[i].append(values[i].detach().cpu())
            self.target_values[i].append(target_values[i].detach().cpu())
            self.values_int[i].append(values_int[i].detach().cpu())
            self.target_values_int[i].append(target_values_int[i].detach().cpu())
            self.h[i].append(h[i].detach().cpu())
            self.c[i].append(c[i].detach().cpu())
            self.h_int[i].append(h_int[i].detach().cpu())
            self.c_int[i].append(c_int[i].detach().cpu())
            self.last_actions[i].append(last_actions[i].detach().cpu())
            self.last_rs_e[i].append(last_rs_e[i].detach().cpu())
            self.last_rs_i[i].append(last_rs_i[i].detach().cpu())

    def reset(self, env_i = None):
        if env_i is None:
            self.lens = [self.m for _ in range(self.num_envs)]
            self.states = [[] for _ in range(self.num_envs)]
            self.actions = [[] for _ in range(self.num_envs)]
            self.rs_e = [[] for _ in range(self.num_envs)]
            self.rs_i = [[] for _ in range(self.num_envs)]
            self.dones = [[] for _ in range(self.num_envs)]
            self.values = [[] for _ in range(self.num_envs)]
            self.target_values = [[] for _ in range(self.num_envs)]
            self.values_int = [[] for _ in range(self.num_envs)]
            self.target_values_int = [[] for _ in range(self.num_envs)]
            self.h = [[] for _ in range(self.num_envs)]
            self.c = [[] for _ in range(self.num_envs)]
            self.h_int = [[] for _ in range(self.num_envs)]
            self.c_int = [[] for _ in range(self.num_envs)]
            self.last_actions = [[] for _ in range(self.num_envs)]
            self.last_rs_e = [[] for _ in range(self.num_envs)]
            self.last_rs_i = [[] for _ in range(self.num_envs)]
        else:
            self.lens[env_i] = self.m
            self.states[env_i] = []
            self.actions[env_i] = []
            self.rs_e[env_i] = []
            self.rs_i[env_i] = []
            self.dones[env_i] = []
            self.values[env_i] = []
            self.target_values[env_i] = []
            self.values_int[env_i] = []
            self.target_values_int[env_i] = []
            self.h[env_i] = []
            self.c[env_i] = []
            self.h_int[env_i] = []
            self.c_int[env_i] = []
            self.last_actions[env_i] = []
            self.last_rs_e[env_i] = []
            self.last_rs_i[env_i] = []

    def get_data(self, env_i = None):
        if env_i is None:
            return self.states, self.actions, self.rs_e, self.rs_i, self.dones, self.values, self.target_values, \
                self.values_int, self.target_values_int, self.h, self.c, self.h_int, self.c_int, self.last_actions, \
                    self.last_rs_e, self.last_rs_i
        return (self.states[env_i], self.actions[env_i], self.rs_e[env_i], self.rs_i[env_i],
                self.dones[env_i], self.values[env_i], self.target_values[env_i], self.values_int[env_i], 
                self.target_values_int[env_i], self.h[env_i], self.c[env_i], self.h_int[env_i], 
                self.c_int[env_i], self.last_actions[env_i], self.last_rs_e[env_i], self.last_rs_i[env_i])