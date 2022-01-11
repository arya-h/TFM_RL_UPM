import gym
import collections
from tensorboardX import SummaryWriter

from TSP.customEnv.envs.TSPagent import TSPAgent
from TSP.customEnv.envs.TSPenv import RegionEnv

ENV_NAME = "customEnv"
GAMMA = 0.9
TEST_EPISODES = 20

if __name__ == "__main__":
    test_env = RegionEnv(5,0)
    agent = TSPAgent(test_env)
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    agent.play_n_random_steps(20)


