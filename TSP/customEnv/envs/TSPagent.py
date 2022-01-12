import collections
import matplotlib.pyplot as plt

GAMMA = 0.9


class TSPAgent():
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        #state has the following format
        #x , y, reward, sequence, done

        #rewards
        self.rewards = collections.defaultdict(float)
        #transitions
        self.transits = {}
        #value
        self.values = collections.defaultdict(float)

#####################################################################################
    def play_n_random_steps(self, count):
        back_home = False
        for _ in range(count):
            action = self.env.action_space.sample()
            #step format is
            # ((self.traveler_x, self.traveler_y), self.reward, self.done, {})
            # determine if city has already been visited
            # in theory shouldnt happen bc agent will only be able to choose between a list of non visited cities
            if action in self.env.sequence:
                if(action==0 and self.env.steps==1):
                    pass
                else:
                    continue

            if (self.env.cities.get(action)['x'] == self.env.traveler_x and self.env.cities.get(action)['y'] == self.env.traveler_y):
                continue

            new_state, reward, is_done, __ = self.env.step(action)
            coord = (self.env.cities.get(self.state)['x'], self.env.cities.get(self.state)['y'])

            self.rewards[(coord, action, new_state)] = reward
            #transits aren't really relevant considering it's deterministic,
            #no probability that we'll take a different path than the one chosen

            self.transits[(coord, action)] = [new_state]
            self.state = self.env.reset() if is_done else new_state

        #plt.show()

#at the end, the rewards dict has the following form (here grouped)
    '''
    ((2, 10), 2, (20, 90)) =82.0
((2, 10), 1, (50, 30)) =52.0
((2, 10), 4, (50, 50)) =62.48199740725323
((2, 10), 3, (90, 30)) =90.24411338142782

((20, 90), 3, (90, 30)) =92.19544457292888
((20, 90), 1, (50, 30)) =67.08203932499369
((20, 90), 4, (50, 50)) =50.0
((20, 90), 0, (2, 10)) =82.0

((90, 30), 4, (50, 50)) =44.721359549995796
((90, 30), 0, (2, 10)) =90.24411338142782
((90, 30), 1, (50, 30)) =40.0
((90, 30), 2, (20, 90)) =92.19544457292888

((50, 30), 0, (2, 10)) =52.0
((50, 30), 2, (20, 90)) =67.08203932499369
((50, 30), 3, (90, 30)) =40.0
((50, 30), 4, (50, 50)) =20.0

((50, 50), 0, (2, 10)) =62.48199740725323
((50, 50), 1, (50, 30)) =20.0
((50, 50), 2, (20, 90)) =50.0
((50, 50), 3, (90, 30)) =44.721359549995796'''
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
        # dict has
        # --KEY : target state
        # --VAL : counter of experienced transitions
        #target_state = self.transits[(state, action)]
        # calculate the sum of all the times the action has been taken from this state
        #total = sum(target_counts.values())
        action_value = 0.0
        coord = (self.env.cities.get(state)['x'], self.env.cities.get(state)['y'])
        # iterate for every target state that the action has landed on
        for src_state, tgt_state in self.transits.items():
            # get the reward for that [s,a,s'] thruple
            if(src_state[0]!=coord):
                 continue
            reward = self.rewards[(coord, action, tgt_state[0])]
            # calculate the updated action value with bellman equation
            # nb, Q(s) =  (probability of landing in that state)*[immediate reward + discounted value for the target state]
            #removed the probability operand, useless here
            action_value += (reward + GAMMA * self.values[tgt_state[0]])
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


            self.transits[(state, action)] = [new_state]
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        state_values = []

        # loop over all states of the environment
        for state in range(self.env.action_space.n):
            #empty the state_values array at the beginning
            state_values.clear()
            coord = self.env.cities.get(state)['x'], self.env.cities.get(state)['y']
            print(state)
            if(state==4):
                print("eccoti qua")
            # for every state we calculate the values of the states
            # reachable from state, which will give us candidates for the value
            # of the state
            for action in range(self.env.action_space.n):
                if action==state:
                    continue
                state_values.append(self.calc_action_value(state,action))

            # update the value of the state with the MIN of the value_action, since
            #we're looking for the minimum path
            self.values[state] = min(state_values)




