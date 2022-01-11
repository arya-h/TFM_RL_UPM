import matplotlib.pyplot as plt

from tsp_bin.environment import RegionScape

if __name__=="__main__":

    env = RegionScape(5)
    obs = env.reset()
    env.step(1)
    env.step(2)
    env.step(3)

    plt.show()

    pass