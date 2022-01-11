import gym
import collections
import matplotlib.pyplot as plt

from TSP.customEnv.envs.TSPenv import RegionEnv
from TSP.main import GAMMA


class TSPAgent():
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        #state has the following format
        #x , y, reward, sequence, done

        #rewards
        self.rewards = collections.defaultdict(float)
        #transitions
        self.transits = collections.defaultdict(collections.Counter)
        #value
        self.values = collections.defaultdict(float)

#####################################################################################
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            #step format is
            # ((self.traveler_x, self.traveler_y), self.reward, self.done, {})
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            #transits aren't really relevant considering it's deterministic,
            #no probability that we'll take a different path than the one chosen
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

        #plt.show()

#####################################################################################
    #action value function

    # calculates the value of the action from the state using
    # --transition table
    # --reward table
    # --values table

    # two purposes
    # 1. select the best action to perform from the state
    # 2. calculate the new value of the state on value iteration.
    def calc_action_value(self, state, action):
        # extract transition counters for the given (state,action) tuple
        # from the transitions table
        # dict has
        # --KEY : target state
        # --VAL : counter of experienced transitions
        target_counts = self.transits[(state, action)]
        # calculate the sum of all the times the action has been taken from this state
        #total = sum(target_counts.values())
        action_value = 0.0
        # iterate for every target state that the action has landed on
        for tgt_state, count in target_counts.items():
            # get the reward for that [s,a,s'] thruple
            reward = self.rewards[(state, action, tgt_state)]
            # calculate the updated action value with bellman equation
            # nb, Q(s) =  (probability of landing in that state)*[immediate reward + discounted value for the target state]
            #removed the probability operand, useless here
            action_value +=  (reward + GAMMA * self.values[tgt_state])
        return action_value

#####################################################################################

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
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        # loop over all states of the environment
        for state in range(self.env.observation_space.n):
            # for every state we calculate the values of the states
            # reachable from state, which will give us candidates for the value
            # of the state
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            # update the value of the state with the maximum of the value_action
            # calculated in the line before
            self.values[state] = max(state_values)




