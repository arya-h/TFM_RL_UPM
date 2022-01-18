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
        self.transits = collections.defaultdict(collections.Counter)
        # value table
        self.values = collections.defaultdict(float)

    # func used to gather random experience, update reward & transitions table
    # there is no need to wait for the end of the entire episode to learn, in stark
    # contrast with cross-entropy, where learning can only happen after full episodes

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()

            new_state, reward, is_done, _ = self.env.step(action)
            as_bytes_state = self.state.tobytes()
            as_bytes_newstate = new_state.tobytes()
            self.rewards[(as_bytes_state, action, as_bytes_newstate)] = reward

            self.transits[(as_bytes_state, action)][as_bytes_newstate] += 1
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
        target_counts = self.transits[(state, action)]
        #calculate the sum of all the times the action has been taken from this state
        total = sum(target_counts.values())
        action_value = 0.0
        #iterate for every target state that the action has landed on
        for tgt_state, count in target_counts.items():
            #get the reward for that [s,a,s'] thruple

            reward = self.rewards[(state, action, tgt_state)]
            #calculate the updated action value with bellman equation
            #nb, Q(s) =  (probability of landing in that state)*[immediate reward + discounted value for the target state]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    # decides the best action to take from a given state
    # it iterates over all possible actions in the env and calculates
    # the value for every action.
    # it returns the action with the largest value, which will be chosen
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
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
        while True:
            # selects the current best action
            state_bytes = state.tobytes()
            action = self.select_action(state_bytes)
            new_state, reward, is_done, _ = env.step(action)
            new_state_bytes = new_state.tobytes()

            self.rewards[(state_bytes, action, new_state_bytes)] = reward
            self.transits[(state_bytes, action)][new_state_bytes] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


    def value_iteration(self):
        # loop over all states of the environment

        #for state in range(self.env.observation_space.shape):
        for state in range(self.env.observation_space.shape[0]):
            #for every state we calculate the values of the states
            #reachable from state, which will give us candidates for the value
            #of the state
            #as_bytes_state = state.tobytes()

            state_values = [self.calc_action_value(state, action)

            for action in range(self.env.action_space.n)]
            #update the value of the state with the maximum of the value_action
            #calculated in the line before
            self.values[state] = max(state_values)

if __name__ == "__main__":
    #test_env = gym.make(ENV_NAME)
    #reduce nodes
    test_env = TSPDistCost()
    agent = AgentTSP()
    #writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = -4000

    iter_no += 1
    # perform N steps to fill reward & transitions tables
    agent.play_n_random_steps(10000)
    print("I have finished playing my random steps")
    # # run value iteration over all states
    print(agent.env.observation_space.shape[0])
    print( type(agent.env.observation_space))
    agent.value_iteration()
    #
    reward = 0.0
    while True:
        for _ in range(TEST_EPISODES):
            # play a full episode, record the sum of the rewards to compare it
            # with next episodes
            reward += agent.play_episode(test_env)
        # next line to get the average reward over the group of episodes
        reward /= TEST_EPISODES
        #writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
            #what is the actual best first path?
            agent.env.plot_network()
            #it converges over the actual worst possible path, not the shortest one


    # writer.close()
