from torch import nn
import torch


class DQN(nn.Module):
    def __init__(self, n_nodes):
        super(DQN, self).__init__()

        self.n_actions = n_nodes
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.5
        self.decay_epsilon = 1500
        self.num_iterations = 2000000
        self.replay_mem_size = 10000
        self.minibatch_size = 16
        #inplace=True means that it will modify the input directly, without allocating any additional output.
        self.relu1 = nn.RReLU(inplace=True)

        self.relu2 = nn.RReLU(inplace=True)

        self.relu3 = nn.RReLU(inplace=True)
        self.fc4   = nn.Linear(in_features=n_nodes+1, out_features=512)
        self.relu4 = nn.RReLU(inplace = True)
        #try to substitute with 1 value out
        self.fc5 = nn.Linear(in_features=512, out_features=n_nodes)

    def forward(self, x):
        floats = []
        for val in x:
            floats.append(val.double())
        #then convert from list to tensor
        tensFloats = torch.stack(floats)
        out = self.relu1(tensFloats)
        out = self.relu2(out)
        out = self.relu3(out)
        out = (out.float())
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out




