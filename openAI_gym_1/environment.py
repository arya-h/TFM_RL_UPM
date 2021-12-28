import random
from typing import List


class Environment:
    def __init__(self):
        #n of steps the agent is allowed to take
        # in the environment
        self.steps_left = 10


    #initialize state to all zeroes
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    #returns to the agent the set of actions it can execute
    #usually the set of actions remains the same, but not
    # every action is possible in some states
    def get_actions(self) -> List[int]:
        return [0, 1]

    #it can detect whether the episode is over
    def is_done(self) -> bool:
        return self.steps_left==0

    #handles the agent's action and returns the reward for this action
    # here the reward is random
    def action(self, action : int) -> float:
        if self.is_done():
            raise Exception("Game over")
        self.steps_left -= 1
        return random.random()



