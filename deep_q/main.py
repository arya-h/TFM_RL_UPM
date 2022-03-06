#guide from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from agent_deep_q import *

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #initialize env
    env = TSPDistCost()
    # agent = Agent(env = env)
    # #
    # agent.play_episodes()
    model = DQN(env.N)
    model.load_state_dict(torch.load("./torch_runs/06_03_2022_Mar0000"))


    agent = Agent(env = env, model = model )
    agent.play_episodes_from_model()



















