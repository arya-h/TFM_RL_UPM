import gym
import collections
from tensorboardX import SummaryWriter
import or_gym
from matplotlib import pyplot as plt
import numpy as np
#example of TSP problem with or-gym's environment
from env_custom import *

ENV_NAME = "TSP-v1"
GAMMA = 0.9
TEST_EPISODES = 20

class AgentTSP:
    def __init__(self):
        #set the environment
        #self.env = or_gym.make(ENV_NAME)
        self.env = TSPDistCost()
        #reset the state
        self.state = self.env.reset()
        # rewards table
        self.rewards = collections.defaultdict(int)
        # transitions table
        self.transits = collections.defaultdict()
        # value table
        self.values = collections.defaultdict(float)

    # func used to gather random experience, update reward & transitions table
    # there is no need to wait for the end of the entire episode to learn, in stark
    # contrast with cross-entropy, where learning can only happen after full episodes

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            state = self.state.tolist()
            state_tuple = tuple(state)
            new_state, reward, is_done, _ = self.env.step(action)
            new_state_tuple = tuple(new_state.tolist())

            as_bytes_newstate = new_state.tobytes()
            self.rewards[(state_tuple, action, new_state_tuple)] = reward

            self.transits[(state_tuple, action)] = new_state_tuple
            self.state = self.env.reset() if is_done else new_state

    #calculates the value of the action from the state using
        # --transition table
        # --reward table
        # --values table

    #two purposes
    # 1. select the best action to perform from the state
    # 2. calculate the new value of the state on value iteration.
    def calc_action_value(self, state, action):
        #extract transition counters for the given (state,action) tuple
        #from the transitions table
        #dict has
        # --KEY : target state
        # --VAL : counter of experienced transitions
        #target_counts = self.transits[(state, action)]
        #calculate the sum of all the times the action has been taken from this state
        #total = sum(target_counts.values())
        action_value = 0.0
        #iterate for every target state that the action has landed on
        #for tgt_state in :
            #get the reward for that [s,a,s'] thruple
            #remove probabilities


        #target state is single, deterministic environment
        tgt_state = self.transits[state, action]

        reward = self.rewards[(state, action, tgt_state)]
        #calculate the updated action value with bellman equation
        #nb, Q(s) =  (probability of landing in that state)*[immediate reward + discounted value for the target state]
        #action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        action_value += (reward + GAMMA * self.values[tgt_state])
        return action_value

    # decides the best action to take from a given state
    # it iterates over all possible actions in the env and calculates
    # the value for every action.
    # it returns the action with the largest value, which will be chosen
    def select_action(self, state):
        #best_value set to 50000 to guarantee it excludes negative values in the if
        best_action, best_value = None, 50000
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value > action_value and action_value>0:
                best_value = action_value
                best_action = action
        return best_action

    # uses the select_action function to choose the best action to take
    # it plays one full episode using the provided environment
    # used to play TEST EPISODES, in order not to influence the current state
    # of the working environment

    # logic : just loop through states

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        #print(f"inside play: {env.state}")
        path = [env.state[0]]

        while True:
            # selects the current best action
            state_tuple = tuple(state.tolist())
            action = self.select_action(state_tuple)
            path.append(action)
            #print(action)
            new_state, reward, is_done, _ = env.step(action)
            new_state_tuple = tuple(new_state.tolist())

            self.rewards[(state_tuple, action, new_state_tuple)] = reward
            self.transits[(state_tuple, action)] = new_state_tuple
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward, path


    def value_iteration(self):
        # loop over all states of the environment

        #for state in range(self.env.observation_space.shape):
        for state,count in self.transits.keys():
            #for every state we calculate the values of the states
            #reachable from state, which will give us candidates for the value
            #of the state
            #as_bytes_state = state.tobytes()

            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]

            #in order to maximize the reward
            #anytime you take an action the reward is
            #if distance = 10, reward = -10
            #negative reward for repeated city : large neg number.
            #update the value of the state with the maximum of the value_action
            #calculated in the line before

            #also try with min
            #select min value that's not negative
            min = 100000
            for val in state_values:
                if val<0:
                    continue
                else:
                    if val < min:
                        min = val

            self.values[state] = min

if __name__ == "__main__":
    #test_env = gym.make(ENV_NAME)
    #reduce nodes
    test_env = TSPDistCost()
    agent = AgentTSP()
    #writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = -4000
    print(agent.env.distance_matrix)

    iter_no += 1
    # perform N steps to fill reward & transitions tables
    agent.play_n_random_steps(50000)
    print("I have finished playing my random steps")
    # # run value iteration over all states
    #print(agent.env.observation_space.shape[0])
    #print( type(agent.env.observation_space))
    agent.value_iteration()
    best_positive_reward=0
    reward = 0.0
    positive=False

    while True:
        newReward, path = agent.play_episode(test_env)
        if(best_reward < 0 and newReward > best_reward and positive==False):
            best_reward = newReward
            if (best_reward > 0) :
                print(f"{newReward} --> {path}")
                agent.env.render(mode="human")
                positive = True
                best_reward = newReward
                continue

        elif newReward < best_reward and newReward>0:
            best_reward = newReward
            print(f"{newReward} --> {path}")
            agent.env.render(mode="human")


    # while True:
    #     newReward, path = agent.play_episode(test_env)
    #     #print(f"{newReward} and path = {path}")
    #     if newReward < best_reward and newReward > 0:
    #         print(f"Path is {path}")
    #         print("Best reward updated %.3f -> %.3f" % (best_reward, newReward))
    #         best_reward = newReward
    #         agent.env.plot_network()

        #print(path)
        # if (newReward <0):
        #     #best_reward = newReward
        #     if(newReward > best_reward):
        #         print(f"Path is {path}")
        #         print("Best reward updated %.3f -> %.3f" % (best_reward, newReward))
        #         best_reward = newReward
        #         # what is the actual best first path?
        #         #agent.env.plot_network()
        #         #print(agent.state)
        # elif (newReward > 0):
        #
        #     if(newReward < best_reward and best_reward):

                # what is the actual best first path?
                #agent.env.plot_network()
                # print(agent.state)
