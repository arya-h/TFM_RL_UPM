from torch import nn
import torch
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_nodes):
        super(DQN, self).__init__()

        self.n_actions = n_nodes
        self.gamma = 0.99
        self.final_epsilon = 0.00001
        self.initial_epsilon = 0.2
        self.decay_epsilon = 2000
        self.replay_mem_size = 20000
        self.minibatch_size = 64
        #inplace=True means that it will modify the input directly, without allocating any additional output.

        # self.relu1 = nn.RReLU(inplace=True)
        #
        # self.relu2 = nn.RReLU(inplace=True)
        #
        # self.relu3 = nn.RReLU(inplace=True)
        # self.fc4   = nn.Linear(in_features=n_nodes+1, out_features=512)
        # self.relu4 = nn.RReLU(inplace = True)
        # #try to substitute with 1 value out
        # self.fc5 = nn.Linear(in_features=512, out_features=n_nodes)

        # FC --> batchnorm -->  SELU  --> dropout
        #layer 1
        self.fc1 = nn.Linear(in_features=n_nodes + 1, out_features=32)
        self.bn1 = nn.LayerNorm(normalized_shape=[32,])
        #self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.RReLU(inplace=True)
        #self.dropout1 = nn.Dropout(p=0.05)
        # layer 2
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.bn2 = nn.LayerNorm(normalized_shape=[64,])
        #self.bn2 = nn.BatchNorm1d(num_features=128)
        self.relu2 = nn.RReLU(inplace=True)
        #self.dropout2 = nn.Dropout(p=0.05)
        # # layer 3
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.bn3 = nn.LayerNorm(normalized_shape=[32,])
        #self.bn3 = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.SELU(inplace=True)
        #self.dropout3 = nn.Dropout(p=0.05)

        self.fc4 = nn.Linear(in_features=32, out_features=n_nodes)

    def forward(self, x):
        floats = []
        for val in x:
            floats.append(val.float())
        #then convert from list to tensor
        tensFloats = torch.stack(floats)

        #layer 1
        out = self.fc1(tensFloats)
        out = self.bn1(out)
        out = self.relu1(out)
        #out = self.dropout1(out)
        #layer2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        #out = self.dropout2(out)

        # #layer 3
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        #out = self.dropout3(out)

        out = self.fc4(out)

        return out




