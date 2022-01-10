from environment import Environment
from agent import Agent
import gym
from typing import TypeVar
import random
from wrappers.actionwrapper import RandomActionWrapper


def random_agent():
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)

    print("TOTAL REWARD : %.4f " % agent.total_reward)

def cart_pole():
    e = gym.make('CartPole-v0')
    #reset the environment, can print out the array of the
    #information about the environment at that moment
    #always remember to reset a newly created environment
    obs = e.reset()
    # print(obs)

    #we can retrieve the initial size of the
    # observation array and the action space in the
    #following way
    #print(e.action_space)
    #prints out Discrete(2). There are two possible actions,
    # 0 -> push to the left // 1 -> push to the right
    #print(e.observation_space)
    #returns Box(4, ) -> vector of size 4. low and high aren't specified,
    # the interval is (-inf, +inf)

    e.step(0)
    #(array([-0.01117676, -0.23552315, -0.00542604,  0.32025903], dtype=float32), 1.0, False, {})
    #1 : new observation array
    #2 : reward
    #3 : done flag
    #4 : extra info, here empty dict

    e.action_space.sample()
    #returns a random sample from the available action space,
    #in this case we only have [0, 1]

    e.observation_space.sample()
    #[ 1.1314398e+00  4.4051259e+37  1.4252383e-01 -3.4471996e+37]
    # random vector of 4 numbers, a plausible array of observation values


#RANDOM CART POLE
def random_cart_pole():
    env = gym.make("CartPole-v0")
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
 total_steps, total_reward))



def wrapper_action_example():
    #pass the wrapper to the environment instance gym.make
    #if i want to nest wrappers i can do so without any constraint on order
    #the wrapper can be used as a normal environment instance

    env = RandomActionWrapper(gym.make("CartPole-v0"))
    #monitor lets us see with a graphical interface the evolution of the agent
    #directory passed as 2nd argumentd should NOT exist
    env = gym.wrappers.Monitor(env, "recording", force=True) #second argument is the directory it will write results to
    total_reward = 0.0
    obs = env.reset() #remember to reset environment at the beginning


    while True:
        obs, reward, done, _ = env.step(0) #takes the same action always,
        total_reward += reward
        if done:
            break
        print("Reward got: %.2f" % total_reward)





if __name__ == '__main__':
    #random_cart_pole()
    wrapper_action_example()

