import collections
import pickle
# example of TSP problem with or-gym's environment
from env_custom import *

ENV_NAME = "TSP-v1"
GAMMA = 0.9
#TEST_EPISODES = 20


class AgentTSP:
    def __init__(self):
        # set the environment
        # self.env = or_gym.make(ENV_NAME)
        self.env = TSPDistCost()
        # reset the state
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
        action_value = 0.0

        # target state is single, deterministic environment
        tgt_state = self.transits[state, action]

        reward = self.rewards[(state, action, tgt_state)]
        # calculate the updated action value with bellman equation
        # nb, Q(s) =  [immediate reward + discounted value for the target state]

        action_value += (reward + GAMMA * self.values[tgt_state])
        return action_value

    # decides the best action to take from a given state
    # it iterates over all possible actions in the env and calculates
    # the value for every action.
    # it returns the action with the best value (according to policy), which will be chosen
    def select_action(self, state):
        # best_value set to 50000 to guarantee it excludes negative values in the if
        best_action, best_value = None, 50000
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value > action_value and action_value > 0:
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
        # print(f"inside play: {env.state}")
        path = [env.state[0]]

        while True:
            # selects the current best action
            state_tuple = tuple(state.tolist())
            action = self.select_action(state_tuple)
            path.append(action)
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
        # in the q-learn variant there isnt a v-function involved,
        # the key in the action-value dictionary is a tuple (s,a)
        for state_action in self.transits.keys():
            same = True
            state = state_action[0]
            state_check = state[1:]
            # exclude impossible states
            # if in the given state all cities have been visited
            cnt = state_check.count(1)
            if cnt == len(state_check):
                continue
            for action in range(self.env.action_space.n):
                action = state_action[1]
                tgt_state = self.transits[state_action]
                tgt_state_check = tgt_state[1:]
                # exclude impossible states
                # if in the given state all cities have been visited
                if(tgt_state_check.count(1)==self.env.N):
                    break
                key = (state, action, tgt_state)
                reward = self.rewards[key]
                best_action = self.select_action(tgt_state)
                val = reward + GAMMA * self.values[(tgt_state, best_action)]
                self.values[(state, action)] = val

def display_distances(matrix, path):
    print("Distances in path :")
    for _ in range(1,len(path)):
        dist = matrix[path[_]][path[_-1]]
        print(f"{path[_-1]} --> {path[_]} : {dist}")

if __name__ == "__main__":

    #test environment for playing episodes
    test_env = TSPDistCost()
    # agent = AgentTSP()
    # iter_no = 0
    #
    #
    # the following lines are commented if we are using the pickled agent

    # iter_no += 1
    # # perform N steps to fill reward & transitions tables
    # agent.play_n_random_steps(200000)
    # print("I have finished playing my random steps")
    # # # run value iteration over all states
    # agent.value_iteration()

    #save data structures with pickle
    # filename = "agent_Qlearn"
    # outfile = open(filename, "wb")
    # pickle.dump(agent, outfile)
    # outfile.close()
    #
    # exit(0)

    infile = open("agent_Qlearn", "rb")
    agent = pickle.load(infile)
    infile.close()
    reward = 0.0
    positive = False
    best_reward = -4000
    # given the path it will determine the single distances with the distance matrix

    while True:
        newReward, path = agent.play_episode(test_env)
        if(best_reward < 0 and newReward > best_reward):
            best_reward = newReward
            if (best_reward > 0) :
                print(f"{newReward} --> {path}")
                #distances can be extracted from the distance matrix
                display_distances(agent.env.distance_matrix, path)
                agent.env.render_custom(path)
                best_reward = newReward
                continue

        elif newReward < best_reward and newReward>0:
            best_reward = newReward
            print(f"{newReward} --> {path}")
            display_distances(agent.env.distance_matrix, path)
            agent.env.render_custom(path)

