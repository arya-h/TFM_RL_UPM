from math import sqrt

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

cities = {
    0 : {
        "id": 0,
        "city": "MAD",
        "x": 2, "y": 10,
        },

   1 :  {"id": 1, "city": "BCN",
     "x": 50, "y": 30,
     },

   2 : {"id": 2, "city": "PRS",
     "x": 20, "y": 90,
     },

   3 : {"id": 3, "city": "TRN",
     "x": 90, "y": 30,
     },

    4 :{"id": 4, "city": "BRLN",
     "x": 50, "y": 50,
     },
}

class RegionEnv(gym.Env):
  metadata = {'render.modes' : ['human']}

  def __init__(self, numCities, startCity):
    super(RegionEnv, self).__init__()
    #observation space
    #it's a matrix

    #contains information about cities, name and coordinates
    self.cities = cities

    #first implementation, but at the end the possible states are just a limited number
    #of coordinates
    self.observation_space = spaces.Box(low=0, high=100, shape=(100, 100, 3))  # matrix
    #second idea, not too sure about this : 1 dimensional array of coordinate tuples,
    #not convinced

    #self.observation_space = Discrete of tuples?  spaces.Tuple(spaces.Discrete(1), spaces.Discrete(1))
    #self.visited = []
    #matrix for pyplot representation
    #matr = np.ones(self.observation_space.shape) * 1
    #self.matrix = plt.matshow(matr)
    #self.matrix = plt.imshow(matr, origin="lower")

    plt.plot()
    plt.axis([0,100,0,100])


    self.numCities = numCities
    self.startCity = startCity
    #set initial coordinates of agent
    self.traveler_x = cities.get(startCity)['x']
    self.traveler_y = cities.get(startCity)['y']
    self.state = startCity

    self.done = False



    #action space. for each step the agent can choose between the cities listed
    self.action_space = spaces.Discrete(numCities)

    #will determine if episode is finished
    self.steps = numCities

    # initial reward
    #self.reward = 0
      
    #current solution
    #initial sequence is empty, i will insert in front of the
    #result sequence the startCity
    self.sequence = []
    #plot
    for val in self.cities.values():
      # plot points and name of city
      plt.scatter(val["x"], val["y"])
      plt.annotate(val["city"], (val["x"], val["y"]))


  def reset(self):
    #in this version state = simply one integer, the id in cities

    self.steps = self.numCities
    #self.reward = 0
    self.done = False
    self.sequence = []
    #self.sequence = [self.startCity]
    self.state = self.startCity
    #self.visited = []
    self.traveler_x = self.cities.get(self.startCity)['x']
    self.traveler_y = self.cities.get(self.startCity)['y']

    # matrix for pyplot representation
    plt.plot()
    plt.axis([0, 100, 0, 100])


    # plot
    for val in self.cities.values():
      # plot points and name of city
      plt.scatter(val["x"], val["y"])
      plt.annotate(val["city"], (val["x"], val["y"]))

    #the returned observed state is the initial coordinates
    return (self.startCity)


  def step(self, action):

    #determine if action is doable
    assert action < self.numCities , "Invalid City ID"

    #add to sequence
    self.sequence.append(action)
    
    #determine distance between points
    dest_x = cities.get(action)['x']
    dest_y = cities.get(action)['y']

    dist = self.calc_dist(self.traveler_x, self.traveler_y, dest_x, dest_y)

    #commented bc it was slowing down the process
    # draw on canvas
    # pt1 = [self.traveler_x, self.traveler_y]
    # pt2 = [dest_x, dest_y]
    #
    # x_val = [pt1[0], pt2[0]]
    # y_val = [pt1[1], pt2[1]]
    #
    # plt.plot(x_val, y_val)

    # set new coords of traveler
    self.traveler_x = dest_x
    self.traveler_y = dest_y

    #traveler has made a trip to a city
    self.steps-=1

    #determine if whole journey is done
    if self.steps==0:
      self.done = True

    #next state is the action : if im in state 0 (MAD) and perform action 3(TRN) my next state
    #will be 3 (TRN)

    return (action, dist, self.done, {})


  def render(self, mode="human", close=False):
    #plot
    plt.show()


    return



  def calc_dist(self, x1, y1, x2, y2):
    dist = sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

    return dist
