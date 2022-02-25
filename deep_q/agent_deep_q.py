from itertools import count

from torch import optim, nn

from custom_env_deep_q import *
from agent_deep_q import *
import torch

from deep_q.main import episode_durations, DQN
from deep_q.replay_memory import Experience, ExperienceBuffer
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_done = 0

class Agent:
    def __init__(self, env : TSPDistCost):
        self.env = env
        self.exp_buffer = ExperienceBuffer(10000)
        self.policy_net = DQN(env.N -1).to(device)
        self.target_net = DQN(env.N -1).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-6)
        self.criterion = nn.HuberLoss()

        # self.rewards_matrix =  env.distance_matrix
        # print(env.distance_matrix)
        # print("##################################à")
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.policy_net.final_epsilon + (self.policy_net.initial_epsilon - self.policy_net.final_epsilon) * math.exp(-1. * steps_done / self.policy_net.decay_epsilon)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #x_np = torch.from_numpy(state)
                #return self.policy_net(state).max(1)[1].view(1, 1)
                index_action = torch.argmax(self.policy_net(state))
                #print(self.policy_net(state).max()[1])
                #return self.policy_net(state).max().view(1)
                return index_action
        else:
            return torch.tensor([[random.randint(0,self.env.N -1 )]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.exp_buffer) < self.policy_net.minibatch_size:
            return
        transitions = self.exp_buffer.sample(self.policy_net.minibatch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        #batch = Experience(*zip(*transitions))
        batch = Experience(*zip(transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.new_state)), device=device, dtype=torch.bool)
        non_final_mask = []
        for val in batch.new_state:
            for subval in val:
                if subval is None:
                    non_final_mask.append(False)
                else:
                    non_final_mask.append(True)

        #convert list of booleans into tensor of dtype bool
        non_final_mask = torch.BoolTensor(non_final_mask)


        print(non_final_mask)
        #noneTensor = torch.ones([self.env.N +1 ], dtype=torch.int32)*-1
        #convert each element in new state to avoid numpy dtype o3bject
        non_final_next_states = []
        for val in batch.new_state:
            for subval in val:
                if subval is not None:
                    non_final_next_states.append(subval)
        # for val in batch.new_state:
        #     if val is not noneTensor:
        #         tensors_newstate.append(torch.from_numpy(val))

        # print(tensors_newstate)

        # non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.new_state
        #                                    if s is not noneTensor])
        #concatenate non final states in the batch in non_final_next_states



        #non_final_next_states = torch.cat([s for s in tensors_newstate])
        #state_batch = torch.cat(batch.state[0])
        state_batch = []
        for val in batch.state:
            for subval in val:
                state_batch.append(subval)
        # action_batch = torch.cat(batch.action)
        action_batch = []
        for val in batch.action:
            for subval in val:
                action_batch.append(subval)

        #reward_batch = torch.cat(batch.reward)
        reward_batch = []
        for val in batch.reward:
            for subval in val:
                reward_batch.append(subval)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        interm = self.policy_net(state_batch)
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        #convert action_batch to tensor
        action_batch = (torch.FloatTensor(action_batch)).type(torch.int64)

        #blocked here, todo  RuntimeError: Index tensor must have the same number of dimensions as input tensor
        #

        state_action_values = interm.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.policy_net.minibatch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max()[0].detach()
        #next_state_values[non_final_mask] = torch.argmax(self.target_net(non_final_next_states))
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.policy_net.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def play_episodes(self):
        num_episodes = 50
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            path = [self.env.state[0]]
            state = self.state
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action)
                path.append(action)
                reward = torch.tensor([reward], device=device)

                # Observe new state

                if not done:
                    next_state = self.env.state
                else:
                    next_state = None#torch.ones([self.env.N + 1], dtype=torch.int32)*-1

                # Store the transition in memory
                self.exp_buffer.append(Experience(state, action,reward, done, next_state))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    episode_durations.append(t + 1)

                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete')
        #self.env.render_custom(path)






