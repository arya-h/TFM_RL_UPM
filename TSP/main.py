import gym
import collections
from tensorboardX import SummaryWriter

from TSP.customEnv.envs.TSPagent import TSPAgent
from TSP.customEnv.envs.TSPenv import RegionEnv

ENV_NAME = "customEnv"
GAMMA = 0.9
TEST_EPISODES = 10

if __name__ == "__main__":
    test_env = RegionEnv(5,0)
    agent = TSPAgent(test_env)
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0


    #agent.play_n_random_steps(300)


    #agent.value_iteration()


    #number of steps must be proportionate to the number of nodes, since it's basically doing all the combinations needed for the transitions dict
    #however, considering that the matrix has the following format (source_state, action) = target_state, but action and target_state are the same
    #playing random steps is a formality in the deterministic model, but fundamental if i introduce probability and randomness like in frozenLake
    agent.play_n_random_steps(500)
    agent.value_iteration()
    reward = 0.0



    for _ in range(TEST_EPISODES):
        reward += agent.play_episode(test_env)
        #print("reward : {}, iter_no: {}".format(reward, _))

    # next line to get the average reward over the group of episodes
    reward /= TEST_EPISODES
    print("reward : {}, iter_no: {}".format(reward))

    print("best path length : {} // sequence : {}".format(agent.bestPath['length'], agent.bestPath['sequence']) )
    
    #draw on canvas
    for _ in range(len(seq)):
        if(_+1 == len(seq)):
            break
        #coord1
        st_1 = seq[_]
        pt1 = (test_env.cities.get(st_1)['x'], test_env.cities.get(st_1)['y'])

        #coord2
        st_2 = seq[_+1]
        pt2 = (test_env.cities.get(st_2)['x'], test_env.cities.get(st_2)['y'])

        x_val = [pt1[0], pt2[0]]
        y_val = [pt1[1], pt2[1]]

        plt.plot(x_val, y_val)

    plt.show()

    #writer.add_scalar("reward", reward, iter_no)
    # if reward > best_reward:
    #     print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
    #     best_reward = reward
    # if reward > -150:
    #     print("Solved in %d iterations!" % iter_no)
    #     break
    #writer.close()

