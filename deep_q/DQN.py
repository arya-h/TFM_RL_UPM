from torch import nn
import torch


class DQN(nn.Module):
    def __init__(self, n_nodes):
        super(DQN, self).__init__()

        self.n_actions = n_nodes
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.decay_epsilon = 150000
        self.num_iterations = 2000000
        self.replay_mem_size = 10000
        self.minibatch_size = 32



        #self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        #inplace=True means that it will modify the input directly, without allocating any additional output.
        self.relu1 = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        #size of output dimension of conv3 by [ (W - K + 2P) / S ] +1 --> [(64 - 3 +2*0) /1] +1 = 62
        # 62*62*64, where 64 are the filters from the prev layer
        self.fc4   = nn.Linear(in_features=8, out_features=512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5   = nn.Linear(in_features=512, out_features=n_nodes)

    def forward(self, x):


        #convert from int to float to be fed to NN
        # floatRes = []
        # for val in x:
        #     floatRes.append(torch.as_tensor(val, dtype=torch.float))
        #
        # tensorFloatRes = torch.FloatTensor(floatRes)

        floats = []
        #here x is a list of tensors. must convert to tensor of tensors (float)
        #first convert the tensors from int to float
        for val in x:
            floats.append(val.double())

        #then convert from list to tensor
        tensFloats = torch.stack(floats)

        out = self.relu1(tensFloats)
        #out = self.conv2(out)
        out = self.relu2(out)
        #out = self.conv3(out)
        out = self.relu3(out)
        #out = out.view(out.size()[0], -1)
        #out = out.view(out, -1)
        out = (out.float())
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out




