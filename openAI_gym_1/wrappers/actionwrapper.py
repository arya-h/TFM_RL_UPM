import gym
from typing import TypeVar
import random

#define the name Action
Action = TypeVar('Action')

#define RandomActionWrapper, extends abstract class ActionWrapper

#

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon = 0.1):
        #initialize the wrapper
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action : Action) -> Action:
        #roll the random dice, if it's < epsilon take the random action
        if random.random() < self.epsilon:
            print("Random action taken")
            #it will take a random action from the sample space
            return self.env.action_space.sample()
        #else it will just return the usual action
        return action