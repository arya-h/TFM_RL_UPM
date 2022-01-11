from math import sqrt

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time
from matplotlib.pyplot import matshow

cities = {
    0 : {
        "state": 0,
        "city": "MAD",
        "x": 2, "y": 10,
        "visited": False},

   1 :  {"state": 1, "city": "BCN",
     "x": 50, "y": 30,
     "visited": False},

   2 : {"state": 2, "city": "PRS",
     "x": 20, "y": 90,
     "visited": False},

   3 : {"state": 3, "city": "TRN",
     "x": 90, "y": 30,
     "visited": False},

    4 :{"state": 4, "city": "BRLN",
     "x": 50, "y": 50,
     "visited": False},
}



font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class RegionScape(Env):
    def __init__(self, numCities=5):
        super(RegionScape, self).__init__()



        # reward
        self.space_traveled = 0  # initially zero when starting trip

        # define the matrix where the traveller can move
        # 3 to add rgb colors
        self.observation_shape = (100, 100, 3)
        self.observation_space = spaces.Discrete(2)  # coordinates of position
        self.action_space = spaces.Discrete(numCities, )  # numCities cities possible to choose from

        self.numCities = numCities  # number of nodes in the map
        # start simple, 5 cities
        self.cities = cities
        #for visual representation
        self.canvas = np.ones(self.observation_shape) * 1
        self.matrix = matshow(self.canvas)
        self.initial_state = 0  # arbitrarily put madrid as first city
        self.traveler_x = self.cities.get(0)["x"]
        self.traveler_y = self.cities.get(0)["y"]

        self.steps = self.numCities  # to reach all cities including the starting one
        matshow(self.canvas)
        for val in self.cities.values():
            # plot points and name of city
            plt.scatter(val["x"], val["y"])
            plt.annotate(val["city"], (val["x"], val["y"]))

        # for text
        self.y_min = 0
        self.x_min = 0
        self.y_max = 100
        self.x_max = 100

        self.text = plt.text(0, -10, "cities left : {}".format(self.steps))

    def reset(self):
        # back to initial configuration
        self.steps = self.numCities
        self.cities = cities
        #reset initial pos
        self.initial_state = 0
        self.traveler_x = self.cities.get(0)["x"]
        self.traveler_y = self.cities.get(0)["y"]
        # reset canvas
        self.canvas = np.ones(self.observation_shape) * 1
        self.matrix = matshow(self.canvas)
        matshow(self.canvas)
        for val in self.cities.values():
            # plot points and name of city
            plt.scatter(val["x"], val["y"])
            plt.annotate(val["city"], (val["x"], val["y"]))

        self.text = self.text.set_text("va bene ok")

    # 2 parts
    # > apply action to agent
    # > other events in env

    def calc_dist(self,x1,y1,x2,y2):
        dist =  sqrt (((x2-x1)**2) + ((y2-y1)**2))

        return dist

    def step(self, action):

        reward = 0
        done = False

        #city chosen
        city_dest = self.cities.get(action)

        # the action index identifies one of the cities.
        assert self.action_space.contains(action), "INVALID ACTION"  # checks if action < 5
        #if it selects a previously visited city give huge 'negative' reward
        if(self.cities.get(action)['visited']):
            reward = +3000

        reward +=  self.calc_dist(self.traveler_x, self.traveler_y,
                                  city_dest["x"], city_dest["y"])


        #draw on canvas
        pt1 = [self.traveler_x, self.traveler_y]
        pt2 = [city_dest['x'], city_dest['y']]

        x_val = [pt1[0], pt2[0]]
        y_val = [pt1[1], pt2[1]]

        plt.plot(x_val, y_val)
        self.text = self.text.set_text( "cities left : {}".format(self.steps))

        self.steps-=1

        #update position of traveller
        self.traveler_x = city_dest['x']
        self.traveler_y = city_dest['y']

        if self.steps == 0:
            done = True

        return
