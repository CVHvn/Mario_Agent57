import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# change from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, (nn.LSTM, nn.LSTMCell)):
        for name, param in layer.named_parameters():
            if "weight" in name:
                torch.nn.init.orthogonal_(param, std)
            elif "bias" in name:
                torch.nn.init.constant_(param, bias_const)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Model(nn.Module):
    def __init__(self, num_policies, input_dim, output_dim, use_layer_init):
        super(Model, self).__init__()
        self.num_policies = num_policies
        self.conv1 = layer_init(nn.Conv2d(1, 32, 8, stride=4)) if use_layer_init else nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2)) if use_layer_init else nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1)) if use_layer_init else nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = layer_init(nn.Linear(3136, 512)) if use_layer_init else nn.Linear(3136, 512)
        self.lstm = layer_init(nn.LSTMCell(512+output_dim+2+self.num_policies, 512)) if use_layer_init else nn.LSTMCell(512+output_dim+2+self.num_policies, 512)
        self.linearV = layer_init(nn.Linear(512, 512)) if use_layer_init else nn.Linear(512, 512)
        self.value = layer_init(nn.Linear(512, 1)) if use_layer_init else nn.Linear(512, 1)
        self.linearA = layer_init(nn.Linear(512, 512)) if use_layer_init else nn.Linear(512, 512)
        self.advantage = layer_init(nn.Linear(512, output_dim)) if use_layer_init else nn.Linear(512, output_dim)

    def forward(self, x, h, c, action, r_e, r_i, j):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        j = F.one_hot(j, num_classes=self.num_policies)
        x = torch.cat([x, action, r_e, r_i, j], -1)
        h, c = self.lstm(x, (h, c))

        a = F.relu(self.linearA(h))
        a = self.advantage(a)

        v = F.relu(self.linearV(h))
        v = self.value(v)

        q = v + a - a.mean(1, keepdim=True)
        return q, h, c

class Feature_Model(nn.Module):
    def __init__(self, input_dim, output_dim, use_layer_init):
        super(Feature_Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(1, 32, 8, stride=4)) if use_layer_init else nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2)) if use_layer_init else nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1)) if use_layer_init else nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = layer_init(nn.Linear(3136, 128)) if use_layer_init else nn.Linear(3136, 128)
        self.linear2 = layer_init(nn.Linear(128, 128)) if use_layer_init else nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Target_Model(nn.Module):
    def __init__(self, input_dim, output_dim, use_layer_init):
        super(Target_Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(1, 32, 8, stride=4)) if use_layer_init else nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2)) if use_layer_init else nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1)) if use_layer_init else nn.Conv2d(64, 64, 3, stride=1)
        self.linear = layer_init(nn.Linear(3136, 128)) if use_layer_init else nn.Linear(3136, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class Embedding_Model(nn.Module):
    def __init__(self, num_action, use_layer_init):
        super(Embedding_Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(1, 32, 8, stride=4)) if use_layer_init else nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2)) if use_layer_init else nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1)) if use_layer_init else nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = layer_init(nn.Linear(3136, 32)) if use_layer_init else nn.Linear(3136, 32)
        self.linear2 = layer_init(nn.Linear(64, 128)) if use_layer_init else nn.Linear(64, 128)
        self.linear3 = layer_init(nn.Linear(128, num_action)) if use_layer_init else nn.Linear(128, num_action)

    def embedding(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        return x

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x