import collections
import sys

GAMMA = 0.9

class TSPAgent():
    def __init__(self, env):
        self.env = env
        self.bestPath  = {
            'length' : sys.maxsize,
            'sequence' : []
        }
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
                continue

            #if im going from state x to state x
            if (self.state == action):
                continue

            # if (self.env.cities.get(action)['x'] == self.env.traveler_x and self.env.cities.get(action)['y'] == self.env.traveler_y):
            #     continue

            new_state, reward, is_done, __ = self.env.step(action)
            coord = (self.env.cities.get(self.state)['x'], self.env.cities.get(self.state)['y'])

            self.rewards[(self.state, action, new_state)] = reward
            #transits aren't really relevant considering it's deterministic,
            #no probability that we'll take a different path than the one chosen

            self.transits[(self.state, action)] = [new_state]
            self.state = self.env.reset() if is_done else new_state

        #plt.show()

#at the end, the rewards dict has the following form (here grouped)
    '''
(2, 3, 3) = 92.19544457292888
(3, 0, 0) = 90.24411338142782
(0, 4, 4) = 62.48199740725323
(4, 1, 1) = 20.0
(1, 2, 2) = 67.08203932499369
(2, 0, 0) = 82.0
(0, 1, 1) = 52.0
(1, 3, 3) = 40.0
(3, 2, 2) = 92.19544457292888
(2, 4, 4) = 50.0
(3, 4, 4) = 44.721359549995796
(0, 3, 3) = 90.24411338142782
(2, 1, 1) = 67.08203932499369
(1, 4, 4) = 20.0
(4, 3, 3) = 44.721359549995796
(3, 1, 1) = 40.0
(1, 0, 0) = 52.0
(0, 2, 2) = 82.0
(4, 2, 2) = 50.0
(4, 0, 0) = 62.48199740725323

#and the transits table
(2, 3) =  [3]
(3, 0) =  [0]
(0, 4) =  [4]
(4, 1) =  [1]
(1, 2) =  [2]
(2, 0) =  [0]
(0, 1) =  [1]
(1, 3) =  [3]
(3, 2) =  [2]
(2, 4) =  [4]
(3, 4) =  [4]
(0, 3) =  [3]
(2, 1) =  [1]
(1, 4) =  [4]
(4, 3) =  [3]
(3, 1) =  [1]
(1, 0) =  [0]
(0, 2) =  [2]
(4, 2) =  [2]
(4, 0) =  [0]

#and the values table
1 =  20.0
2 =  50.0
3 =  44.721359549995796
4 =  38.0
0 =  52.0

'''
    
    
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
        # --VAL : counter of experienced transitions (will explore in branch 'randomness')
            tgt_state = self.transits[(state,action)]
            reward = self.rewards[(state, action, tgt_state[0])]
            action_value = (reward + GAMMA * self.values[tgt_state[0]])
            return action_value

#####################################################################################

    # decides the best action to take from a given state
    # it iterates over all possible actions in the env and calculates
    # the value for every action.
    # it returns the action with the lowest value (since we are aiming to minimize the length
    # of the path), which will be chosen
    def select_action(self, state):
        best_action, best_value = None, None


        for action in range(self.env.action_space.n):

            if (action in self.env.sequence):
                continue

            #if im going from state x to state x
            if(state==action):
                continue


            if(action == self.env.startCity and self.env.steps!=1):
                continue


            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value > action_value:
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
            #excludes actions already selected
            if(action in self.env.sequence):
                continue
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward

            self.transits[(state, action)] = [new_state]
            total_reward += reward
            if is_done:
                break
            state = new_state

        if(total_reward < self.bestPath.get('length')):
            self.bestPath['length'] = total_reward
            self.bestPath['sequence'] = self.env.sequence
        return total_reward

    def value_iteration(self):
        state_values = []

        # loop over all states of the environment
        for state in range(self.env.action_space.n):
            #empty the state_values array at the beginning
            state_values.clear()
            coord = self.env.cities.get(state)['x'], self.env.cities.get(state)['y']
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




