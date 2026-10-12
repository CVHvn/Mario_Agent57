from PIL import Image
from collections import deque
from datetime import datetime
import pickle
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#import multiprocessing as mp
from torchvision import transforms as T
import gc

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.environment import *
from src.episode_memory import *
from src.sumtree import *
from src.per import *
from src.model import *
from src.ucb import *
from src.running_mean_std import *

class Agent():
    def __init__(self, world, stage, action_type, envs, num_envs, num_policies, additional_bonus_state_8_4_option,
                 state_dim, action_dim, save_dir, save_model_step, save_figure_step, learn_step, total_step_or_episode, 
                 total_step, total_episode, model, target_model, model_int, target_model_int, rnd_target_model, 
                 rnd_predict_model, embedding_model, gammas, betas, learning_rate, learning_rate_int, embedding_wd, 
                 adam_eps, max_grad_norm, target_update_freq, replay_buffer_size, per, per_eps, per_alpha, per_beta, 
                 eta, m, l, n, batch_size, loss_type, epsilons, start_learning_step, start_learning_sequence, device):
        self.world = world
        self.stage = stage
        self.action_type = action_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.learn_step = learn_step
        self.total_step_or_episode = total_step_or_episode
        self.total_step = total_step
        self.total_episode = total_episode
        self.target_update_freq = target_update_freq

        self.current_step = 0
        self.current_episode = 0

        self.save_model_step = save_model_step
        self.save_figure_step = save_figure_step

        self.device = device
        self.save_dir = save_dir

        self.num_envs = num_envs
        self.num_policies = num_policies
        self.envs = envs
        self.additional_bonus_state_8_4_option = additional_bonus_state_8_4_option
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.model_int = model_int.to(self.device)
        self.target_model_int = target_model_int.to(self.device)

        self.rnd_target_model = rnd_target_model.to(self.device)
        self.rnd_predict_model = rnd_predict_model.to(self.device)
        self.embedding_model = embedding_model.to(self.device)

        self.learning_rate = learning_rate
        self.learning_rate_int = learning_rate_int
        self.embedding_wd = embedding_wd
        self.adam_eps = adam_eps
        self.gammas = gammas.to(device).float()
        self.betas = betas.to(device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps = self.adam_eps)
        self.optimizer_int = torch.optim.Adam(self.model_int.parameters(), lr=self.learning_rate, eps = self.adam_eps)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predict_model.parameters(), lr=self.learning_rate_int, eps = self.adam_eps)
        self.embedding_optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=self.learning_rate_int, eps = self.adam_eps, 
                                                    weight_decay = self.embedding_wd)

        #self.model = torch.compile(self.model)

        self.m = m
        self.l = l
        self.n = n
        self.epsilons = epsilons
        self.start_learning_step = start_learning_step
        self.start_learning_sequence = start_learning_sequence

        self.per_eps = per_eps
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.eta = eta
        self.replay_buffer_size = replay_buffer_size
        self.per = per

        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.is_completed = False

        self.env = None
        self.max_test_score = -1e9
        self.loss_type = loss_type

        # I just log 1000 lastest update and print it to log.
        self.losses = np.zeros((1000,)).reshape(-1)
        self.losses_int = np.zeros((1000,)).reshape(-1)
        self.loss_index = 0
        self.len_loss = 0

        self.rewards_int = np.zeros((1000,)).reshape(-1)
        self.rewards_int_pseudo_count = np.zeros((1000,)).reshape(-1)
        self.rewards_int_rnd = np.zeros((1000,)).reshape(-1)
        self.rewards_int_rnd_norm = np.zeros((1000,)).reshape(-1)
        self.rewards_int_idx = 0
        self.rewards_int_len = 0

        self.embedding_loss = np.zeros((1000,)).reshape(-1)
        self.rnd_loss = np.zeros((1000,)).reshape(-1)
        self.embedding_loss_idx = 0
        self.embedding_loss_len = 0

        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84), device=self.device)
        self.dm_rms = RunningMeanStd(shape=(self.num_envs,))
        self.rnd_rms = RunningMeanStd(shape=(1,))

    def save_figure(self, is_training = False):
        # test current model and save model/figure if model yield best total rewards.
        # create env for testing, reset test env
        if self.env is None:
            self.env = create_env(self.world, self.stage, self.action_type, self.additional_bonus_state_8_4_option, True)
        state = self.env.reset()
        done = False

        images = []
        total_r_e = 0
        total_r_i = 0
        total_step = 0
        num_repeat_action = 0
        old_action = -1

        episode_time = datetime.now()

        # create h, c as zeros
        h = torch.zeros((1, 512), dtype=torch.float, device = self.device)
        c = torch.zeros((1, 512), dtype=torch.float, device = self.device)

        last_actions = torch.zeros((1, self.action_dim), device = self.device)
        last_rs_e = torch.zeros((1,), device = self.device).reshape(-1)
        last_rs_i = torch.zeros((1,), device = self.device).reshape(-1)

        state_ = torch.tensor(np.array(state), device = self.device)
        state_ = state_.reshape(1, 1, state_.shape[1], state_.shape[2])
        state_ = ((state_ - self.obs_rms.mean.to(self.device)) / torch.sqrt(self.obs_rms.var.to(self.device))).clip(-5, 5)

        with torch.no_grad():
            embeddings = self.embedding_model.embedding(state_)

        episodic_memorys = [[embeddings[0]]]

        # play 1 episode, just get loop action with max probability from model until the episode end.
        while not done:
            with torch.no_grad():
                Q, h, c = self.model(torch.tensor(state, dtype = torch.float, device = self.device).unsqueeze(0), h, c,
                                     last_actions, last_rs_e.unsqueeze(-1), last_rs_i.unsqueeze(-1), torch.zeros((1, ), dtype = torch.long).to(self.device))

            action = Q.argmax(-1).item()
            next_state, r_e, done, trunc, info = self.env.step(action)
            state = next_state

            state_ = torch.tensor(np.array(state), device = self.device)
            state_ = state_.reshape(1, 1, state_.shape[1], state_.shape[2])
            state_ = ((state_ - self.obs_rms.mean.to(self.device)) / torch.sqrt(self.obs_rms.var.to(self.device))).clip(-5, 5)

            rewards_int_pseudo_count, embeddings = self.compute_reward_int_pseudo_counts(state_, episodic_memorys, is_test=True)
            rewards_int_rnd = self.compute_reward_int_rnd(state_)

            rewards_int_rnd_norm = (rewards_int_rnd - self.rnd_rms.mean.cpu().numpy()) / (np.sqrt(self.rnd_rms.var.cpu().numpy()) + 1e-11)
            rewards_int_rnd_norm = 1 + rewards_int_rnd_norm.astype(np.float32)

            r_i = np.maximum(rewards_int_rnd_norm, np.ones_like(rewards_int_rnd_norm))
            r_i = np.minimum(r_i, np.ones_like(r_i) * 5.)
            r_i = r_i * rewards_int_pseudo_count

            img = Image.fromarray(self.env.current_state)
            images.append(img)
            total_r_e += r_e
            total_r_i += r_i.item()
            total_step += 1

            last_actions = F.one_hot(torch.as_tensor(np.array([action]), device = self.device), self.action_dim).to(self.device)
            last_rs_e[0] = r_e
            last_rs_i[0] = r_i.item()

            episodic_memorys[0].append(embeddings[0])

            if action == old_action:
                num_repeat_action += 1
            else:
                num_repeat_action = 0
            old_action = action
            if num_repeat_action == 200:
                break

        #logging, if model yield better result, save figure (test_episode.mp4) and model (best_model.pth)
        if is_training:
            f_out = open(f"logging_test.txt", "a")
            f_out.write(f'episode_reward: {total_r_e:.4f} int_reward: {total_r_i:.4f} episode_step: {total_step} '
                        f'current_step: {self.current_step} loss: {(self.losses.sum()/self.len_loss):.4f} '
                        f'loss_int: {(self.losses_int.sum()/self.len_loss):.4f} '
                        f'loss_embedding: {(self.embedding_loss.sum()/self.embedding_loss_len):.4f} '
                        f'loss_rnd: {(self.rnd_loss.sum()/self.embedding_loss_len):.4f} '
                        f'reward_int: {(self.rewards_int.sum()/self.rewards_int_len):.4f} '
                        f'reward_int_pseudo_count: {(self.rewards_int_pseudo_count.sum()/self.rewards_int_len):.4f} '
                        f'reward_int_rnd: {(self.rewards_int_rnd.sum()/self.rewards_int_len):.4f} '
                        f'reward_int_rnd_norm: {(self.rewards_int_rnd_norm.sum()/self.rewards_int_len):.4f} '
                        f'len_per: {self.per.real_size} episode_time: {datetime.now() - episode_time}\n')
            f_out.close()

        if total_r_e > self.max_test_score or info['flag_get']:
            imageio.mimsave('test_episode.mp4', images)
            self.max_test_score = total_r_e
            if is_training:
                torch.save(self.model.state_dict(), f"best_model.pth")
                torch.save(self.model_int.state_dict(), f"best_model_int.pth")
                torch.save(self.rnd_target_model.state_dict(), f"best_rnd_target_model.pth")
                torch.save(self.rnd_predict_model.state_dict(), f"best_rnd_predict_model.pth")
                torch.save(self.embedding_model.state_dict(), f"best_embedding_model.pth")
                with open("best_rms.pickle", "wb") as f:
                    pickle.dump({'obs_rms_mean': self.obs_rms.mean, 
                                 'obs_rms_var': self.obs_rms.var, 
                                 'dm_rms_mean': self.dm_rms.mean, 
                                 'dm_rms_var': self.dm_rms.var, 
                                 'rnd_rms_mean': self.rnd_rms.mean, 
                                 'rnd_rms_var': self.rnd_rms.var}, f)

        # if model can complete this game, stop training by set self.is_completed to True
        if info['flag_get']:
            self.is_completed = True

    def save_model(self):
        torch.save(self.model.state_dict(), f"model_{self.current_step}.pth")
        torch.save(self.model_int.state_dict(), f"model_int_{self.current_step}.pth")
        torch.save(self.rnd_target_model.state_dict(), f"rnd_target_model_{self.current_step}.pth")
        torch.save(self.rnd_predict_model.state_dict(), f"rnd_predict_model_{self.current_step}.pth")
        torch.save(self.embedding_model.state_dict(), f"embedding_model_{self.current_step}.pth")
        with open(f"rms_{self.current_step}.pickle", "wb") as f:
            pickle.dump({'obs_rms_mean': self.obs_rms.mean, 
                         'obs_rms_var': self.obs_rms.var, 
                         'dm_rms_mean': self.dm_rms.mean, 
                         'dm_rms_var': self.dm_rms.var, 
                         'rnd_rms_mean': self.rnd_rms.mean, 
                         'rnd_rms_var': self.rnd_rms.var}, f)

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = {"model": f"model_{self.current_step}.pth",
                          "model_int": f"model_int_{self.current_step}.pth",
                          "rnd_target_model": f"rnd_target_model_{self.current_step}.pth",
                          "rnd_predict_model": f"rnd_predict_model_{self.current_step}.pth",
                          "embedding_model": f"embedding_model_{self.current_step}.pth",
                          "rms": f"rms_{self.current_step}.pickle"}
        self.model.load_state_dict(torch.load(model_path["model"]))
        self.model_int.load_state_dict(torch.load(model_path["model_int"]))
        self.rnd_target_model.load_state_dict(torch.load(model_path["rnd_target_model"]))
        self.rnd_predict_model.load_state_dict(torch.load(model_path["rnd_predict_model"]))
        self.embedding_model.load_state_dict(torch.load(model_path["embedding_model"]))
        with open(model_path["rms"], "rb") as f:
            data = pickle.load(f)
            self.obs_rms.mean = data['obs_rms_mean']
            self.obs_rms.var = data['obs_rms_var']
            self.dm_rms.mean = data['dm_rms_mean']
            self.dm_rms.var = data['dm_rms_var']
            self.rnd_rms.mean = data['rnd_rms_mean']
            self.rnd_rms.var = data['rnd_rms_var']

    def update_loss_statis(self, loss, loss_int):
        # update loss for logging, just save 1000 latest updates.
        self.losses[self.loss_index] = loss
        self.losses_int[self.loss_index] = loss_int
        self.loss_index = (self.loss_index + 1)%1000
        self.len_loss = min(self.len_loss+1, 1000)

    def update_reward_int_log(self, rewards_int_pseudo_count, rewards_int_rnd, rewards_int_rnd_norm, rewards_int):
        for i in range(len(rewards_int)):
            self.rewards_int[self.rewards_int_idx] = rewards_int[i].mean()
            self.rewards_int_pseudo_count[self.rewards_int_idx] = rewards_int_pseudo_count[i].mean()
            self.rewards_int_rnd[self.rewards_int_idx] = rewards_int_rnd[i].mean()
            self.rewards_int_rnd_norm[self.rewards_int_idx] = rewards_int_rnd_norm[i].mean()
            self.rewards_int_idx = (self.rewards_int_idx + 1)%1000
            self.rewards_int_len = min(self.rewards_int_len+1, 1000)

    def update_ngu_log(self, loss_emb, loss_rnd):
        self.embedding_loss[self.embedding_loss_idx] = loss_emb
        self.rnd_loss[self.embedding_loss_idx] = loss_rnd
        self.embedding_loss_idx = (self.embedding_loss_idx + 1)%1000
        self.embedding_loss_len = min(self.embedding_loss_len+1, 1000)

    def select_action(self, states, h, c, target_h, target_c, h_int, c_int, target_h_int, target_c_int, 
                      last_actions, last_rs_e, last_rs_i, policies_idx):
        states = torch.as_tensor(np.array(states), device = self.device)

        with torch.no_grad():
            Q, h, c = self.model(states, h, c, last_actions, 
                                 last_rs_e.unsqueeze(-1), last_rs_i.unsqueeze(-1),
                                 policies_idx)
            target_q, target_h, target_c = self.target_model(states, target_h, target_c, last_actions, 
                                                             last_rs_e.unsqueeze(-1), last_rs_i.unsqueeze(-1),
                                                             policies_idx)

            Q_int, h_int, c_int = self.model_int(states, h_int, c_int, last_actions, 
                                             last_rs_e.unsqueeze(-1), last_rs_i.unsqueeze(-1),
                                             policies_idx)
            target_q_int, target_h_int, target_c_int = self.target_model_int(states, target_h_int, target_c_int, last_actions, 
                                                                last_rs_e.unsqueeze(-1), last_rs_i.unsqueeze(-1),
                                                                policies_idx)

            actions = (self.inverse_value_rescale(Q) + 
                       self.betas[policies_idx].reshape(-1, 1) * self.inverse_value_rescale(Q_int)).argmax(-1) # need check shape

            n_actions = Q.shape[1]
            batch_size = states.shape[0]
            random_actions = torch.randint(0, n_actions, (batch_size,), device=self.device)

            use_random = torch.rand(batch_size, device=self.device) < self.epsilons
            actions = torch.where(use_random, random_actions, actions)

        return actions, Q, target_q, Q_int, target_q_int, h, c, target_h, target_c, h_int, c_int, target_h_int, target_c_int

    def update_target_model(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.copy_(online_param)
            for target_param, online_param in zip(self.target_model_int.parameters(), self.model_int.parameters()):
                target_param.copy_(online_param)
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def value_rescale(value, eps=1e-3):
        return value.sign()*((value.abs()+1).sqrt()-1) + eps*value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4*eps*(value.abs()+1+eps)).sqrt() - 1) / (2*eps)
        return value.sign() * (temp.square() - 1)

    def learn(self):
        # get all data
        # states: (m+l+n) x b x 1 x 84 x 84
        # actions: (m+l+n) x b
        # rewards: (m+l+n) x b
        # dones: (m+l+n) x b
        # init_h: b x 512
        # init_c: b x 512
        # last_actions: b x 512 x a
        # last_rewards: b x 512
        # masks: (m+l+n) x b
        (states, actions, rs_e, rs_i, dones, init_h, init_c, init_h_int, init_c_int, last_actions, 
            last_rs_e, last_rs_i, policies_idx, masks), weights, sample_idx = self.per.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rs_e = torch.as_tensor(rs_e, device=self.device, dtype=torch.float32)
        rs_i = torch.as_tensor(rs_i, device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.float32)
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        last_actions = torch.as_tensor(last_actions, device=self.device)
        last_rs_e = torch.as_tensor(last_rs_e, device=self.device)
        last_rs_i = torch.as_tensor(last_rs_i, device=self.device)
        policies_idx = torch.as_tensor(policies_idx, device=self.device)

        Qs, target_qs = [], []
        Qs_int, target_qs_int = [], []

        # burn in
        init_h, init_c = torch.tensor(init_h).to(self.device), torch.tensor(init_c).to(self.device)
        h, c = init_h.clone(), init_c.clone()
        target_h, target_c = init_h.clone(), init_c.clone()

        init_h_int, init_c_int = torch.tensor(init_h_int).to(self.device), torch.tensor(init_c_int).to(self.device)
        h_int, c_int = init_h_int.clone(), init_c_int.clone()
        target_h_int, target_c_int = init_h_int.clone(), init_c_int.clone()

        for t in range(self.m):
            with torch.no_grad():
                q, h, c = self.model(states[t], h, c, last_actions[t], last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
                target_q, target_h, target_c = self.target_model(states[t], target_h, target_c, last_actions[t], 
                                                                 last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
                h, c = h * masks[t].unsqueeze(1), c * masks[t].unsqueeze(1)
                target_h, target_c = target_h * masks[t].unsqueeze(1), target_c * masks[t].unsqueeze(1)
                Qs.append(q)
                target_qs.append(target_q)

                q_int, h_int, c_int = self.model_int(states[t], h_int, c_int, last_actions[t], last_rs_e[t].unsqueeze(-1), 
                                                 last_rs_i[t].unsqueeze(-1), policies_idx)
                target_q_int, target_h_int, target_c_int = self.target_model_int(states[t], target_h_int, target_c_int, last_actions[t], 
                                                                 last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
                h_int, c_int = h_int * masks[t].unsqueeze(1), c_int * masks[t].unsqueeze(1)
                target_h_int, target_c_int = target_h_int * masks[t].unsqueeze(1), target_c_int * masks[t].unsqueeze(1)
                Qs_int.append(q_int)
                target_qs_int.append(target_q_int)

        # calculate Q and target_Q
        for t in range(self.m, self.m + self.l):
            q, h, c = self.model(states[t], h, c, last_actions[t], last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
            with torch.no_grad():
                target_q, target_h, target_c = self.target_model(states[t], target_h, target_c, last_actions[t], 
                                                                 last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
            Qs.append(q)
            target_qs.append(target_q)

            q_int, h_int, c_int = self.model_int(states[t], h_int, c_int, last_actions[t], last_rs_e[t].unsqueeze(-1), 
                                             last_rs_i[t].unsqueeze(-1), policies_idx)
            with torch.no_grad():
                target_q_int, target_h_int, target_c_int = self.target_model_int(states[t], target_h_int, target_c_int, last_actions[t], 
                                                                 last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
            Qs_int.append(q_int)
            target_qs_int.append(target_q_int)

        # last n_step
        h = h.detach().clone()
        c = c.detach().clone()

        h_int = h_int.detach().clone()
        c_int = c_int.detach().clone()

        with torch.no_grad():
            for t in range(self.m + self.l, self.m + self.l + self.n):
                q, h, c = self.model(states[t], h, c, last_actions[t], last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
                target_q, target_h, target_c = self.target_model(states[t], target_h, target_c, last_actions[t], 
                                                                 last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
                Qs.append(q)
                target_qs.append(target_q)

                q_int, h_int, c_int = self.model_int(states[t], h_int, c_int, last_actions[t], last_rs_e[t].unsqueeze(-1), 
                                                     last_rs_i[t].unsqueeze(-1), policies_idx)
                target_q_int, target_h_int, target_c_int = self.target_model_int(states[t], target_h_int, target_c_int, last_actions[t], 
                                                                 last_rs_e[t].unsqueeze(-1), last_rs_i[t].unsqueeze(-1), policies_idx)
                Qs_int.append(q_int)
                target_qs_int.append(target_q_int)

        # calculate target
        idx = torch.arange(0, len(h), device = self.device)
        targets = []
        targets_int = []
        with torch.no_grad():
            for t in range(self.m + self.l - 1, self.m-1, -1):
                max_action = (self.inverse_value_rescale(Qs[t + self.n]) + 
                              self.betas[policies_idx.reshape(-1)].reshape(-1, 1) * self.inverse_value_rescale(Qs_int[t + self.n])).argmax(-1) # need check shape
                target = self.inverse_value_rescale(target_qs[t + self.n][idx, max_action])
                target_int = self.inverse_value_rescale(target_qs_int[t + self.n][idx, max_action])

                for k in range(t + self.n - 1, t-1, -1):
                    target = target * self.gammas[policies_idx].reshape(-1) * (1. - dones[k]) + rs_e[k]
                    target_int = target_int * self.gammas[policies_idx].reshape(-1) * (1. - dones[k]) + rs_i[k]
                targets.append(self.value_rescale(target))
                targets_int.append(self.value_rescale(target_int))

        # calculate loss
        loss, loss_int = 0., 0.
        targets = targets[::-1]
        targets_int = targets_int[::-1]
        max_delta = torch.zeros((len(idx),)).reshape(-1).to(self.device)
        mean_delta = 0.
        max_delta_int = torch.zeros((len(idx),)).reshape(-1).to(self.device)
        mean_delta_int = 0.
        counts = 0.
        for t in range(self.m, self.m + self.l):
            if self.loss_type == 'huber':
                delta = (F.huber_loss(Qs[t][idx, actions[t]], targets[t-self.m], reduction = "none") * masks[t])
                delta_int = (F.huber_loss(Qs_int[t][idx, actions[t]], targets_int[t-self.m], reduction = "none") * masks[t])
            else:
                delta = (F.mse_loss(Qs[t][idx, actions[t]], targets[t-self.m], reduction = "none") * masks[t])
                delta_int = (F.mse_loss(Qs_int[t][idx, actions[t]], targets_int[t-self.m], reduction = "none") * masks[t])
            loss += (delta.reshape(-1) * weights.reshape(-1)).sum()
            loss_int += (delta_int.reshape(-1) * weights.reshape(-1)).sum()
            counts += masks[t]
            with torch.no_grad():
                delta = (Qs[t][idx, actions[t]] - targets[t-self.m]).abs() * masks[t]
                delta_int = (Qs_int[t][idx, actions[t]] - targets_int[t-self.m]).abs() * masks[t]
                max_delta = torch.maximum(max_delta, delta.reshape(-1))
                mean_delta = mean_delta + delta.reshape(-1)
                max_delta_int = torch.maximum(max_delta_int, delta_int.reshape(-1))
                mean_delta_int = mean_delta_int + delta_int.reshape(-1)
        mean_delta = mean_delta / counts
        mean_delta_int = mean_delta_int / counts
        loss /= counts.sum()
        loss_int /= counts.sum()

        #update model
        self.optimizer.zero_grad()
        self.optimizer_int.zero_grad()
        loss.backward()
        loss_int.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model_int.parameters(), self.max_grad_norm)
        self.optimizer_int.step()
        self.optimizer.step()

        #update loss statis
        self.update_loss_statis(loss.item(), loss_int.item())

        #update priority
        priorities = self.eta * max_delta + (1-self.eta) * mean_delta
        priorities_int = self.eta * max_delta_int + (1-self.eta) * mean_delta_int
        self.per.update_priorities(sample_idx, (priorities + priorities_int) * 0.5)

        masks = masks.bool()
        idx = torch.zeros_like(masks, dtype = torch.bool)
        idx_ = torch.zeros_like(masks, dtype = torch.bool)

        rev = masks[self.m:self.m+self.l].flip(0)
        cumsum = rev.cumsum(dim=0)
        idx[self.m:self.m+self.l] = (rev & (cumsum >= 2) & (cumsum <= 6)).flip(0)
        
        idx_[self.m+1:self.m+self.l+1] = idx[self.m:self.m+self.l]

        actions = actions[idx]
        next_states = states[idx_].reshape(-1, 1, states.shape[-2], states.shape[-1])
        states = states[idx].reshape(-1, 1, states.shape[-2], states.shape[-1])

        actions = actions.reshape(-1)
        states = ((states - self.obs_rms.mean.to(self.device)) / torch.sqrt(self.obs_rms.var.to(self.device))).clip(-5, 5)
        next_states = ((next_states - self.obs_rms.mean.to(self.device)) / torch.sqrt(self.obs_rms.var.to(self.device))).clip(-5, 5)
        
        loss_rnd = self.train_rnd(next_states)
        loss_embedding = self.train_embedding(states, next_states, actions)

        self.update_ngu_log(loss_embedding, loss_rnd)

    def calculate_target(self, r_i, r_e, dones, values, target_values, values_int, target_values_int, gamma, beta):
        targets = []
        targets_int = []
        T = len(r_i)
        for t in range(T):
            if t + self.n < T:
                max_action = (self.inverse_value_rescale(values[t + self.n]) + 
                              beta * self.inverse_value_rescale(values_int[t + self.n])).argmax(-1)
                target = self.inverse_value_rescale(target_values[t + self.n][max_action])
                target_int = self.inverse_value_rescale(target_values_int[t + self.n][max_action])
            else:
                target = 0
                target_int = 0

            for k in range(min(T-1, t + self.n - 1), t-1, -1):
                target = target * gamma * (1. - dones[k].float()) + r_e[k]
                target_int = target_int * gamma * (1. - dones[k].float()) + r_i[k]
            targets.append(self.value_rescale(target))
            targets_int.append(self.value_rescale(target_int))

        return torch.tensor(targets).reshape(-1), torch.tensor(targets_int).reshape(-1)

    def calculate_priorities(self, values, targets, actions, sequence_idx):
        T = len(values)
        priorities = []
        idxs = torch.arange(0, len(values), device = self.device)
        values = values[idxs, actions]
        deltas = (values.cpu() - targets).abs()
        for i, idx in enumerate(sequence_idx):
            start = idx
            end = min(T, idx + self.l)
            delta = deltas[start:end]
            max_delta = delta.max()
            mean_delta = delta.mean()
            priorities.append(self.eta * max_delta.item() + (1-self.eta) * mean_delta.item())
        return torch.as_tensor(np.array(priorities))

    def save_episode(self, policy_idx, data):
        # (T x 1 x 84 x 84) (T) (T) (T) (T a) (T a) (T 512) (T 512) (T 512 a) (T 512) (1)
        states, actions, rs_e, rs_i, dones, values, target_values, values_int, \
            target_values_int, h, c, h_int, c_int, last_actions, last_rs_e, last_rs_i = data

        states = np.array(states)
        actions, rs_e, rs_i, dones = torch.tensor(actions, dtype=torch.long).to(self.device), \
                        torch.tensor(rs_e).to(self.device), torch.tensor(rs_i).to(self.device), torch.tensor(dones).to(self.device)
        values, target_values = torch.stack(values, 0).to(self.device), torch.stack(target_values, 0).to(self.device)
        h, c = torch.stack(h, 0).to(self.device), torch.stack(c, 0).to(self.device)
        values_int, target_values_int = torch.stack(values_int, 0).to(self.device), torch.stack(target_values_int, 0).to(self.device)
        h_int, c_int = torch.stack(h_int, 0).to(self.device), torch.stack(c_int, 0).to(self.device)
        last_actions, last_rs_e, last_rs_i = np.array(last_actions), np.array(last_rs_e), np.array(last_rs_i)

        T = len(states)
        sequence_idx = torch.arange(0, T, self.l).reshape(-1) # (S)
        burnin_sequence_idx = torch.clamp(sequence_idx - self.m, min=0)
        end_sequence_idx = torch.clamp(sequence_idx + self.l + self.n - 1, max=T - 1)

        targets, targets_int = self.calculate_target(rs_e, rs_i, dones, values, target_values, values_int, target_values_int, 
                                                     self.gammas[policy_idx], self.betas[policy_idx]) # (T 512)
        priorities = self.calculate_priorities(values, targets, actions, sequence_idx) # (S)
        priorities_int = self.calculate_priorities(values_int, targets_int, actions, sequence_idx) # (S)

        priorities = (priorities+priorities_int)*0.5
        priorities = priorities.numpy().tolist()

        h, c = h[burnin_sequence_idx].detach(), c[burnin_sequence_idx].detach()
        h_int, c_int = h_int[burnin_sequence_idx].detach(), c_int[burnin_sequence_idx].detach()

        self.per.add((states, actions.cpu(), rs_e.cpu(), rs_i.cpu(), dones.cpu(), h.cpu(), c.cpu(), 
                      h_int.cpu(), c_int.cpu(), last_actions, last_rs_e, last_rs_i),
                     burnin_sequence_idx, sequence_idx, end_sequence_idx, priorities, policy_idx)
        
    def compute_reward_int_rnd(self, next_state):
        with torch.no_grad():
            target_features = self.rnd_target_model(next_state)
            features = self.rnd_predict_model(next_state)
        rewards_int = ((features - target_features)**2).mean(-1)
        return rewards_int.reshape(-1).cpu().numpy()

    def compute_reward_int_pseudo_counts(self, next_states, episodic_memorys, k=10,
                                                                                kernel_cluster_distance=0.008,
                                                                                kernel_epsilon=0.0001,
                                                                                c=0.001,
                                                                                sm=8,
                                                                                is_test=False):
        with torch.no_grad():
            embeddings = self.embedding_model.embedding(next_states)

        lengths = [len(episodic_memory) for episodic_memory in episodic_memorys]
        max_l, min_l = max(k, max(lengths)), min(lengths)
        episodic_memorys = torch.stack([torch.cat([torch.stack(episodic_memory, 0),
                                       torch.stack([embedding] * (max_l + 1 - length), 0)], 0) for episodic_memory, embedding, length in zip(episodic_memorys, embeddings, lengths)],
                                       0)[:, :-1]
        masks = torch.zeros((episodic_memorys.shape[0], max_l), device = self.device)
        for i, length in enumerate(lengths):
            masks[i][length:] = 1e9
        embeddings = embeddings.unsqueeze(1)

        dist = ((embeddings - episodic_memorys)**2).mean(-1) + masks
        dist = torch.topk(dist, dim=1, k=k, largest=False)[0]
        masks = (dist > 1e8-1).float()
        dist = dist * (masks < 1).float()

        # moving average (not sure it correct)
        dm = dist.sum(-1) / torch.maximum((masks < 1).float().sum(-1), torch.ones(episodic_memorys.shape[0], device = self.device))
        if not is_test:
            self.dm_rms.update(dm.cpu().reshape(1, -1))
            dist = dist / (self.dm_rms.mean.reshape(-1, 1).to(self.device) + 1e-9)
        else:
            dist = dist / (dm.reshape(-1, 1) + 1e-9) # dont use moving average

        dist = torch.maximum(dist - kernel_cluster_distance, torch.zeros_like(dist, device = self.device))
        kernel = kernel_epsilon / (dist + kernel_epsilon)
        kernel = kernel * (masks < 1).float()
        s = torch.sqrt(kernel.sum(-1)) + c

        idxes = (s.isnan()) | (s > sm)
        s[idxes] = 0
        s[torch.logical_not(idxes)] = 1 / s[torch.logical_not(idxes)]
        return s.cpu().numpy(), embeddings.reshape(embeddings.shape[0], -1)

    def train_embedding(self, states, next_states, action_index):

        labels = F.one_hot(action_index, self.action_dim).float().to(self.device)

        self.embedding_optimizer.zero_grad()

        logits = self.embedding_model(states, next_states)

        loss_embedding = F.cross_entropy(logits, labels)
        loss_embedding.backward()
        torch.nn.utils.clip_grad_norm_(self.embedding_model.parameters(), self.max_grad_norm)
        self.embedding_optimizer.step()

        return loss_embedding.item()
    
    def train_rnd(self, next_states):
        self.rnd_optimizer.zero_grad()
        with torch.no_grad():
            target_features = self.rnd_target_model(next_states)
        features = self.rnd_predict_model(next_states)
        loss_rnd = ((features - target_features)**2).mean()
        loss_rnd.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd_predict_model.parameters(), self.max_grad_norm)
        self.rnd_optimizer.step()
        return loss_rnd.item()

    def train(self):
        episode_reward = [0] * self.num_envs
        episode_step = [0] * self.num_envs
        max_episode_reward = 0
        max_episode_step = 0
        episode_time = [datetime.now() for _ in range(self.num_envs)]
        total_time = datetime.now()

        last_episode_rewards = []

        #reset envs
        states = self.envs.reset() #list n, 1, 84, 84

        # create h, c as zeros
        h = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512
        c = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512

        target_h = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512
        target_c = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512

        h_int = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512
        c_int = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512

        target_h_int = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512
        target_c_int = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512

        episode_buffer = Episode_memory(self.num_envs, self.m, self.l, self.n)

        self.update_target_model()

        last_actions = torch.zeros((self.num_envs, self.action_dim), device = self.device) # n a
        last_rs_e = torch.zeros((self.num_envs,), device = self.device).reshape(-1) # n
        last_rs_i = torch.zeros((self.num_envs,), device = self.device).reshape(-1) # n

        self.ucbs = [UCB(self.num_policies) for _ in range(self.num_envs)]
        #worker_idx = torch.concat([torch.arange(self.num_policies, device = self.device)] * (self.num_envs//self.num_policies)) # n
        worker_idx = torch.zeros((self.num_envs,), dtype = torch.int64, device = self.device) #n
        for i in range(self.num_envs):
            worker_idx[i] = self.ucbs[i].get_idx()

        # update obs rms and normalize state
        states_ = torch.tensor(np.array(states), device = self.device)
        states_ = states_.reshape(-1, 1, states_.shape[2], states_.shape[3])
        self.obs_rms.update(states_.cpu())
        states_ = ((states_ - self.obs_rms.mean.to(self.device)) / torch.sqrt(self.obs_rms.var.to(self.device))).clip(-5, 5)

        with torch.no_grad():
            embeddings = self.embedding_model.embedding(states_)

        del states_
        gc.collect()
        torch.cuda.empty_cache()

        episodic_memorys = [[embeddings[i]] for i in range(self.num_envs)]

        while True:
            # finish training if agent reach total_step or total_episode base on what type of total_step_or_episode is step or episode
            self.current_step += 1

            if self.total_step_or_episode == 'step':
                if self.current_step >= self.total_step:
                    break
            else:
                if self.current_episode >= self.total_episode:
                    break

            # (n) (n x a), (n x a), (n x 512), (n x 512), (n x 512), (n x 512)
            actions, values, target_values, values_int, target_values_int, \
                new_h, new_c, new_target_h, new_target_c, new_h_int, new_c_int, \
                    new_target_h_int, new_target_c_int = self.select_action(states, h, c, target_h, target_c, h_int, c_int, 
                                                                            target_h_int, target_c_int, last_actions, last_rs_e,
                                                                            last_rs_i, worker_idx)

            # (n, 1, 84, 84) (n, 1, 84, 84) (n,) (n,)
            next_states, next_states_, rs_e, dones, truncs, infos = self.envs.step(actions.cpu().numpy())

            next_states = torch.tensor(np.array(next_states), device = self.device)
            next_states = next_states.reshape(-1, 1, next_states.shape[2], next_states.shape[3])
            self.obs_rms.update(next_states.cpu())
            next_states = ((next_states - self.obs_rms.mean.to(self.device)) / torch.sqrt(self.obs_rms.var.to(self.device))).clip(-5, 5)

            rewards_int_pseudo_count, embeddings = self.compute_reward_int_pseudo_counts(next_states, episodic_memorys)
            rewards_int_rnd = self.compute_reward_int_rnd(next_states)

            self.rnd_rms.update(torch.tensor(rewards_int_rnd).reshape(-1))
            rewards_int_rnd_norm = (rewards_int_rnd - self.rnd_rms.mean.cpu().numpy()) / (np.sqrt(self.rnd_rms.var.cpu().numpy()) + 1e-11)
            rewards_int_rnd_norm = 1 + rewards_int_rnd_norm.astype(np.float32)

            rs_i = np.maximum(rewards_int_rnd_norm, np.ones_like(rewards_int_rnd_norm))
            rs_i = np.minimum(rs_i, np.ones_like(rs_i) * 5.)
            rs_i = rs_i * rewards_int_pseudo_count

            self.update_reward_int_log(rewards_int_pseudo_count, rewards_int_rnd, rewards_int_rnd_norm, rs_i)

            # save to episode memorys
            episode_buffer.save(states, actions, rs_e, rs_i, dones, values, target_values, values_int, target_values_int,
                                h, c, h_int, c_int, last_actions, last_rs_e, last_rs_i)

            last_actions = F.one_hot(actions.long().reshape(-1),
                                     self.action_dim).to(self.device).float()
            last_rs_e = torch.as_tensor(np.array(rs_e), device=self.device).reshape(-1).float()
            last_rs_i = torch.as_tensor(np.array(rs_i), device=self.device).reshape(-1).float()

            episode_reward = [x + reward for x, reward in zip(episode_reward, rs_e)]
            episode_step = [x+1 for x in episode_step]

            # reset h and c to zeros for enviroments that just ending episode, just multiply h and c with (1-dones)
            h = new_h * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            c = new_c * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

            target_h = new_target_h * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            target_c = new_target_c * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

            h_int = new_h_int * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            c_int = new_c_int * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

            target_h_int = new_target_h_int * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            target_c_int = new_target_c_int * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

             # logging after each step, if 1 episode is ending, I will log this to logging.txt
            for i, done in enumerate(dones):
                episodic_memorys[i].append(embeddings[i])
                if done:
                    self.current_episode += 1
                    max_episode_reward = max(max_episode_reward, episode_reward[i])
                    max_episode_step = max(max_episode_step, episode_step[i])
                    last_episode_rewards.append(episode_reward[i])
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f'episode: {self.current_episode} agent: {i} policy: {worker_idx[i]} epsilon: {self.epsilons[i]:.4f} '
                                f'rewards: {episode_reward[i]:.4f} steps: {episode_step[i]} complete: {infos[i]["flag_get"]==True} mean_rewards: '
                                f'{np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean():.4f} '
                                f'max_rewards: {max_episode_reward:.4f} max_steps: {max_episode_step} current_step: {self.current_step} '
                                f'loss: {(self.losses.sum()/self.len_loss):.4f} loss_int: {(self.losses_int.sum()/self.len_loss):.4f} '
                                f'loss_embedding: {(self.embedding_loss.sum()/self.embedding_loss_len):.4f} '
                                f'loss_rnd: {(self.rnd_loss.sum()/self.embedding_loss_len):.4f} reward_int: '
                                f'{(self.rewards_int.sum()/self.rewards_int_len):.4f} reward_int_pseudo_count: '
                                f'{(self.rewards_int_pseudo_count.sum()/self.rewards_int_len):.4f} reward_int_rnd: '
                                f'{(self.rewards_int_rnd.sum()/self.rewards_int_len):.4f} reward_int_rnd_norm: '
                                f'{(self.rewards_int_rnd_norm.sum()/self.rewards_int_len):.4f} len_per: {self.per.real_size} '
                                f'episode_time: {datetime.now() - episode_time[i]} total_time: {datetime.now() - total_time}\n')
                    f_out.close()
                    self.ucbs[i].push(worker_idx[i].item(), episode_reward[i])
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_time[i] = datetime.now()

                    self.save_episode(worker_idx[i].item(), episode_buffer.get_data(i))
                    episode_buffer.reset(i)

                    last_actions[i] = last_actions[i] * 0
                    last_rs_e[i] = 0
                    last_rs_i[i] = 0
                    episodic_memorys[i] = [embeddings[i]]
                    worker_idx[i] = self.ucbs[i].get_idx()

            # training agent every learn_step
            if (self.current_step % self.learn_step == 0 and
                self.current_step * self.num_envs >= self.start_learning_step and
                self.per.real_size >= self.start_learning_sequence and
                self.per.real_size > self.batch_size):
                self.learn()

            # update target model every target_update_freq
            if self.current_step % (self.target_update_freq * self.learn_step) == 0:
                self.update_target_model()

            # eval agent every save_figure_step
            if self.current_step % self.save_figure_step == 0:
                self.save_figure(is_training=True)
                if self.is_completed:
                    return
                gc.collect()
                torch.cuda.empty_cache()

            if self.current_step % self.save_model_step == 0:
                self.save_model()

            states = list(next_states_)

        f_out = open(f"logging.txt", "a")
        f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} max_rewards: {max_episode_reward} \
max_steps: {max_episode_step} current_step: {self.current_step} total_time: {datetime.now() - total_time}\n')
        f_out.close()