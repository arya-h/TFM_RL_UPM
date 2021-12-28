from environment import *

class Agent:
    def __init__(self):
        #counter that keeps track of the reward accumulated
        # throughout the episode
        self.total_reward = 0.0


    def step(self,  env : Environment):
        #observe the environment
        current_obs = env.get_observation()
        #make a decision based on the observations made
        actions = env.get_actions()
        # take the action in the environment and
        # get the reward for the current step
        reward = env.action(random.choice(actions))
        self.total_reward += reward