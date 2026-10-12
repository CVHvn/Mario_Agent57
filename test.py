import argparse
import numpy as np
import torch

from src.agent import *
from src.environment import *
from src.episode_memory import *
from src.sumtree import *
from src.per import *
from src.model import *
from src.ucb import *
from src.running_mean_std import *

def get_args():
    parser = argparse.ArgumentParser(
        """Agent57 implement to playing Super Mario Bros""")
    parser.add_argument("--pretrained_model", type=str, default="1-1", help = 'Pretrained model path')

    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=32, help='Number of environment, paper use 256')
    parser.add_argument('--num_policies', type=int, default=32, help='Number of policies')
    parser.add_argument('--learn_step', type=int, default=4, help='Number of steps between training model, paper use 52')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size, paper use 64')

    parser.add_argument('--gamma0', type=float, default=0.9999, help='Use to calculate discount factor for rewards')
    parser.add_argument('--gamma1', type=float, default=0.997, help='Use to calculate discount factor for rewards')
    parser.add_argument('--gamma2', type=float, default=0.99, help='Use to calculate discount factor for rewards')

    parser.add_argument('--beta_min', type=float, default=0., help='Use to calculate intrinsic reward coef')
    parser.add_argument('--beta_max', type=float, default=0.3, help='Use to calculate intrinsic reward coef')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of intrinsic and extrinsic value models')
    parser.add_argument('--learning_rate_int', type=float, default=5e-4, help='Learning rate of other models')
    parser.add_argument('--embedding_wd', type=float, default=1e-5, help='Weight decay of embedding model')
    parser.add_argument('--adam_eps', type=float, default=1e-4, help='Eps of Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=40, help='Max gradient norm')

    parser.add_argument("--loss_type", type=str, default="mse", help = "Use mse or huber loss")

    parser.add_argument('--target_update_freq', type=int, default=1500, help='num training step between update target model')
    parser.add_argument('--replay_buffer_size', type=int, default=int(5e4), help='per size, number sequences in per')
    parser.add_argument('--replay_buffer_sample_size', type=int, default=int(4e6)+125, help='per sample size, number samples in per')
    
    parser.add_argument('--per_eps', type=float, default=1e-2, help='epsilon in per')
    parser.add_argument('--per_alpha', type=float, default=0.9, help='alpha in per')
    parser.add_argument('--per_beta', type=float, default=0., help='beta in per')
    parser.add_argument('--eta', type=float, default=0.9, help='eta in per, use to balance between mean and max sample errors when calculate priority for sequence')
    
    parser.add_argument('--start_learning_step', type=int, default=50000, help='start training after start_learning_step/num_envs steps')
    parser.add_argument('--start_learning_sequence', type=int, default=6250, help='start training after per have start_learning_sequence sequences')

    parser.add_argument('--total_step', type=int, default=int(1e7), help='Total step for training')
    parser.add_argument('--save_model_step', type=int, default=int(1e5), help='Number of steps between saving model')
    parser.add_argument('--save_figure_step', type=int, default=400, help='Number of steps between testing model')
    parser.add_argument('--total_step_or_episode', type=str, default='step', help='choice stop training base on total step or total episode')
    parser.add_argument('--total_episode', type=int, default=None, help='Total episodes for training')

    parser.add_argument("--action_dim", type=int, default=12, help='12 if set action_type to complex else 7')
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--state_dim", type=tuple, default=(1, 84, 84))

    parser.add_argument("--m", type=int, default=40, help='burn-in step')
    parser.add_argument("--l", type=int, default=80, help='sequence length')
    parser.add_argument("--n", type=int, default=5, help='n-steps')

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--use_layer_init", type=bool, default=True, help = 'Use layer init or not')
    parser.add_argument("--additional_bonus_state_8_4_option", type=str, default="no", help = 'Option to add more bonus reward for state 8-4, this value can set to no (Do not add more reward), right_pipe (Add +50 bonus reward when Mario go to right pipe)')
    
    args = parser.parse_args()
    return args

def test(config):
    model = Model(config.num_policies, config.state_dim, config.action_dim, config.use_layer_init)
    model_int = Model(config.num_policies, config.state_dim, config.action_dim, config.use_layer_init)
    target_model = Model(config.num_policies, config.state_dim, config.action_dim, config.use_layer_init)
    target_model_int = Model(config.num_policies, config.state_dim, config.action_dim, config.use_layer_init)

    rnd_target_model = Target_Model(config.state_dim, config.action_dim, config.use_layer_init)
    rnd_predict_model = Feature_Model(config.state_dim, config.action_dim, config.use_layer_init)
    embedding_model = Embedding_Model(config.action_dim, config.use_layer_init)

    epsilon_base = 0.4
    epsilons = torch.tensor(np.array([epsilon_base ** (1 + i / (config.num_envs - 1) * 8) for i in range(config.num_envs)]),
                            device = "cuda" if torch.cuda.is_available() else "cpu")

    N = config.num_policies
    gammas1 = [config.gamma0 + (config.gamma1 - config.gamma0) * torch.sigmoid(torch.tensor([10 * (2*i - 6)/6])).item() for i in range(1, 7)]
    gammas2 = [1 - np.exp(((N - i - 1) * np.log(1 - config.gamma1) + (i-8) * np.log(1 - config.gamma2)) / (N-9)) for i in range(8, N)]
    gammas = torch.tensor([config.gamma0] + gammas1 + [config.gamma1] + gammas2)

    N = config.num_policies
    if N >= 16:
        betas = torch.tensor([config.beta_min] + (config.beta_max * torch.sigmoid(torch.tensor([10 * (2*i - (N-2)) / (N-2) for i in range(N-2)]))).numpy().tolist() + [config.beta_max])
    else:
        betas = torch.arange(config.beta_min, config.beta_max+(config.beta_max - config.beta_min)/(N-1), (config.beta_max - config.beta_min)/(N-1))
    
    agent = Agent(world = config.world, stage = config.stage, action_type = config.action_type, envs = None, num_envs = config.num_envs,
              num_policies = config.num_policies, additional_bonus_state_8_4_option = config.additional_bonus_state_8_4_option,
              state_dim = config.state_dim, action_dim = config.action_dim, save_dir = config.save_dir,
              save_model_step = config.save_model_step, save_figure_step = config.save_figure_step, learn_step = config.learn_step,
              total_step_or_episode = config.total_step_or_episode, total_step = config.total_step, total_episode = config.total_episode,
              model = model, target_model = target_model, model_int = model_int, target_model_int = target_model_int, 
              rnd_target_model = rnd_target_model, rnd_predict_model = rnd_predict_model, 
              embedding_model = embedding_model, gammas = gammas, betas = betas, learning_rate = config.learning_rate,
              learning_rate_int = config.learning_rate_int, embedding_wd = config.embedding_wd, adam_eps = config.adam_eps,
              max_grad_norm = config.max_grad_norm, target_update_freq = config.target_update_freq, replay_buffer_size = config.replay_buffer_size,
              per = None, batch_size = config.batch_size, loss_type = config.loss_type, per_eps = config.per_eps, per_alpha = config.per_alpha,
              per_beta = config.per_beta, eta = config.eta, m = config.m, l = config.l, n = config.n, epsilons = epsilons,
              start_learning_step = config.start_learning_step, start_learning_sequence = config.start_learning_sequence,
              device = "cuda" if torch.cuda.is_available() else "cpu")
    
    agent.load_model({"model": f"{config.pretrained_model}\\best_model.pth",
                      "model_int": f"{config.pretrained_model}\\best_model_int.pth",
                      "rnd_target_model": f"{config.pretrained_model}\\best_rnd_target_model.pth",
                      "rnd_predict_model": f"{config.pretrained_model}\\best_rnd_predict_model.pth",
                      "embedding_model": f"{config.pretrained_model}\\best_embedding_model.pth",
                      "rms": f"{config.pretrained_model}\\best_rms.pickle"})
    agent.save_figure()

if __name__ == "__main__":
    config = get_args()
    test(config)
