# Edit from [Howuhh PER](https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py). Use Gemini 3.5 Flash and Gemini 3.1 Pro to improve memory usage for episode saving.

from src.sumtree import *
import numpy as np
import random

class R2D2ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=100000, sample_size=4000000, l=40, m=40, n=5, eps=1e-2, alpha=0.6, beta=0.4):
        self.tree = SumTree(size=buffer_size)

        self.action_dim = action_dim

        # PER Hyperparameters
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        self.l = l  # Sequence length for training (40 steps)
        self.m = m  # Burn-in length for init lstm (40 steps)
        self.n = n  # N-step lookahead to calculate Target Q
        self.seq_len = m + l + n

        # per size
        self.sample_size = int(sample_size)  # 4e6 observations
        self.size = int(buffer_size)        # 1e5 sequences

        self.states = np.zeros((self.sample_size, *state_dim), dtype=np.uint8)
        self.actions = np.zeros((self.sample_size,), dtype=np.int64)
        self.rs_e = np.zeros((self.sample_size,), dtype=np.float32)
        self.rs_i = np.zeros((self.sample_size,), dtype=np.float32)
        self.dones = np.zeros((self.sample_size,), dtype=np.float32)

        self.last_actions = np.zeros((self.sample_size, self.action_dim), dtype=np.float32)
        self.last_rs_e = np.zeros((self.sample_size,), dtype=np.float32)
        self.last_rs_i = np.zeros((self.sample_size,), dtype=np.float32)

        self.hs = np.zeros((self.size, 512), dtype=np.float32)
        self.cs = np.zeros((self.size, 512), dtype=np.float32)
        self.hs_int = np.zeros((self.size, 512), dtype=np.float32)
        self.cs_int = np.zeros((self.size, 512), dtype=np.float32)
        self.policies_idx = np.zeros((self.size, ), dtype=np.int64)

        self.burnin_idx = np.zeros((self.size,), dtype=np.int64)
        self.start_idx = np.zeros((self.size,), dtype=np.int64)
        self.end_idx = np.zeros((self.size,), dtype=np.int64)

        self.global_sample_count = 0  
        self.count = 0                
        self.real_size = 0            

        self.active = np.zeros((self.size,), dtype=np.bool_)
        self.active_size = 0

        self.allocate_mem()

    def allocate_mem(self):
        self.states += 0
        self.actions += 0
        self.rs_e += 0
        self.rs_i += 0
        self.dones += 0
        self.hs += 0
        self.cs += 0
        self.hs_int += 0
        self.cs_int += 0
        self.last_actions += 0
        self.last_rs_e += 0
        self.last_rs_i += 0
        self.policies_idx += 0

    def invalidate_overwritten_sequences(self):
        min_valid_global_idx = max(0, self.global_sample_count - self.sample_size)

        invalid = self.active & (self.burnin_idx < min_valid_global_idx)
        invalid_idxs = np.flatnonzero(invalid)

        for idx in invalid_idxs:
            self.tree.update(int(idx), 0.0)

        self.active[invalid_idxs] = False
        self.active_size -= len(invalid_idxs)

    def add(self, transition, burnin_idx_arr, start_idx_arr, end_idx_arr, priorities, policies_idx):
        state, action, r_e, r_i, done, h, c, h_int, c_int, last_actions, last_r_e, last_r_i = transition
        episode_len = len(state)

        flat_start = self.global_sample_count % self.sample_size
        flat_end = flat_start + episode_len

        if flat_end <= self.sample_size:
            self.states[flat_start:flat_end] = state
            self.actions[flat_start:flat_end] = action
            self.rs_e[flat_start:flat_end] = r_e
            self.rs_i[flat_start:flat_end] = r_i
            self.dones[flat_start:flat_end] = done
            self.last_actions[flat_start:flat_end] = last_actions
            self.last_rs_e[flat_start:flat_end] = last_r_e
            self.last_rs_i[flat_start:flat_end] = last_r_i
        else:
            first_part_len = self.sample_size - flat_start

            self.states[flat_start:] = state[:first_part_len]
            self.actions[flat_start:] = action[:first_part_len]
            self.rs_e[flat_start:] = r_e[:first_part_len]
            self.rs_i[flat_start:] = r_i[:first_part_len]
            self.dones[flat_start:] = done[:first_part_len]
            self.last_actions[flat_start:] = last_actions[:first_part_len]
            self.last_rs_e[flat_start:] = last_r_e[:first_part_len]
            self.last_rs_i[flat_start:] = last_r_i[:first_part_len]

            second_part_len = episode_len - first_part_len
            self.states[:second_part_len] = state[first_part_len:]
            self.actions[:second_part_len] = action[first_part_len:]
            self.rs_e[:second_part_len] = r_e[first_part_len:]
            self.rs_i[:second_part_len] = r_i[first_part_len:]
            self.dones[:second_part_len] = done[first_part_len:]
            self.last_actions[:second_part_len] = last_actions[first_part_len:]
            self.last_rs_e[:second_part_len] = last_r_e[first_part_len:]
            self.last_rs_i[:second_part_len] = last_r_i[first_part_len:]

        for i in range(len(start_idx_arr)):

            if self.active[self.count]:
                self.active[self.count] = False
                self.active_size -= 1

            self.tree.add(self.max_priority, self.count)
            self.update_priorities([self.count], [priorities[i]])

            self.active[self.count] = True
            self.active_size += 1

            self.burnin_idx[self.count] = self.global_sample_count + int(burnin_idx_arr[i])
            self.start_idx[self.count] = self.global_sample_count + int(start_idx_arr[i])
            self.end_idx[self.count] = self.global_sample_count + int(end_idx_arr[i])

            self.hs[self.count] = h[i]
            self.cs[self.count] = c[i]
            self.hs_int[self.count] = h_int[i]
            self.cs_int[self.count] = c_int[i]
            self.policies_idx[self.count] = policies_idx

            self.count = (self.count + 1) % self.size
            self.real_size = min(self.size, self.real_size + 1)

        self.global_sample_count += episode_len
        self.invalidate_overwritten_sequences()

    def sample(self, batch_size):
        assert self.active_size >= batch_size, "Buffer chưa tích lũy đủ số lượng chuỗi tối thiểu."
        assert self.tree.total > 0

        sample_idxs, tree_idxs = [], []
        priorities = np.zeros((batch_size,))

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            while True:
                a, b = segment * i, segment * (i + 1)
                cumsum = random.uniform(a, b)
                tree_idx, priority, sample_idx = self.tree.get(cumsum)

                if not self.active[sample_idx]:
                    continue

                priorities[i] = priority
                tree_idxs.append(tree_idx)
                sample_idxs.append(sample_idx)
                break

        probs = priorities / (self.tree.total + 1e-8)
        weights = (self.active_size * probs) ** -self.beta
        weights = weights / (weights.max() + 1e-8)

        base_idx_raw = self.start_idx[sample_idxs] - self.m
        base_idx = base_idx_raw % self.sample_size
        gather_idx = (base_idx[:, None] + np.arange(self.seq_len)[None, :]) % self.sample_size

        states  = self.states[gather_idx]
        actions = self.actions[gather_idx]
        rs_e = self.rs_e[gather_idx]
        rs_i = self.rs_i[gather_idx]
        dones = self.dones[gather_idx]
        hs = self.hs[sample_idxs]
        cs = self.cs[sample_idxs]
        hs_int = self.hs_int[sample_idxs]
        cs_int = self.cs_int[sample_idxs]
        last_actions = self.last_actions[gather_idx]
        last_rs_e = self.last_rs_e[gather_idx]
        last_rs_i = self.last_rs_i[gather_idx]
        policies_idx = self.policies_idx[sample_idxs]

        masks = np.ones((batch_size, self.seq_len), dtype=np.float32)
        pos = np.arange(self.seq_len)

        valid_burnin = self.start_idx[sample_idxs] - self.burnin_idx[sample_idxs]
        missing_burnin = np.maximum(0, self.m - valid_burnin)
        masks[pos[None, :] < missing_burnin[:, None]] = 0

        valid_tail = self.end_idx[sample_idxs] - self.start_idx[sample_idxs] + 1
        missing_tail = np.maximum(0, (self.l + self.n) - valid_tail)
        masks[pos[None, :] >= (self.seq_len - missing_tail)[:, None]] = 0

        batch = (
            np.swapaxes(states, 0, 1),
            np.swapaxes(actions, 0, 1),
            np.swapaxes(rs_e, 0, 1),
            np.swapaxes(rs_i, 0, 1),
            np.swapaxes(dones, 0, 1),
            hs, cs, hs_int, cs_int,
            np.swapaxes(last_actions, 0, 1),
            np.swapaxes(last_rs_e, 0, 1),
            np.swapaxes(last_rs_i, 0, 1),
            policies_idx,
            np.swapaxes(masks, 0, 1)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (float(priority) + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)