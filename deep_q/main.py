#guide from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
from itertools import count

import torch
from torch import optim
import random

from custom_env_deep_q import *
from DQN import *
from replay_memory import *
from agent_deep_q import *

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPS_DECAY = 150000
EPS_START = 1.0
EPS_END = 0.01



episode_durations = []




if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #initialize env
    env = TSPDistCost()
    agent = Agent(env = env)

    agent.play_episodes()

















