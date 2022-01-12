import gym
import collections
from tensorboardX import SummaryWriter

from TSP.customEnv.envs.TSPagent import TSPAgent
from TSP.customEnv.envs.TSPenv import RegionEnv

ENV_NAME = "customEnv"
GAMMA = 0.9
TEST_EPISODES = 5

if __name__ == "__main__":
    test_env = RegionEnv(5,0)
    agent = TSPAgent(test_env)
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0

    agent.play_n_random_steps(300)


    agent.value_iteration()
    test_env.render()
    agent.play_episode(test_env)

    # while True:
    #     agent.play_n_random_steps(10)
    #     agent.value_iteration()
    #     reward = 0.0
    #
    #     for _ in range(TEST_EPISODES):
    #         reward += agent.play_episode(test_env)
    #
    #     # next line to get the average reward over the group of episodes
    #     reward /= TEST_EPISODES
    #     writer.add_scalar("reward", reward, iter_no)
    #     if reward > best_reward:
    #         print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
    #         best_reward = reward
    #     if reward > -150:
    #         print("Solved in %d iterations!" % iter_no)
    #         break
    #     writer.close()

