import numpy as np
import matplotlib.pyplot as plt

from TSP.environment import RegionScape

if __name__=="__main__":

    env = RegionScape(5)
    obs = env.reset()

    env.step(1)
    plt.show()

    pass