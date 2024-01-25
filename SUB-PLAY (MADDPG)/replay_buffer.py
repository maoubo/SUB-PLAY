import numpy as np
import torch
import pickle

class ReplayBuffer:
    def __init__(self, args, obs_dim, act_dim, buffer_type):
        self.args = args
        self.memory_capacity = args.memory_capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = args.device
        self.pointer = 0
        if buffer_type == 0:
            self.num_agents = args.num_victim
        else:
            self.num_agents = args.num_attacker
        self.buffer = {'state': np.zeros([self.memory_capacity, self.num_agents, self.obs_dim[0]]),
                       'act': np.zeros([self.memory_capacity, self.num_agents, self.act_dim[0]]),
                       'reward': np.zeros([self.memory_capacity, self.num_agents, 1]),
                       'state_': np.zeros([self.memory_capacity, self.num_agents, self.obs_dim[0]]),
                       'done': np.zeros([self.memory_capacity, self.num_agents, 1])}
        self.batch = {'state': torch.zeros([args.batch_size, self.num_agents, self.obs_dim[0]]),
                      'act': torch.zeros([args.batch_size, self.num_agents, self.act_dim[0]]),
                      'reward': torch.zeros([args.batch_size, self.num_agents, 1]),
                      'state_': torch.zeros([args.batch_size, self.num_agents, self.obs_dim[0]]),
                      'done': torch.zeros([args.batch_size, self.num_agents, 1])}

    def store_transition(self, state, act, reward, state_, done):
        pointer = self.pointer % self.memory_capacity
        self.buffer['state'][pointer] = state
        if self.args.scenario == 'world_comm':
            for i in range(self.num_agents):
                self.buffer['act'][pointer][i][:self.act_dim[i]] = act[i][:self.act_dim[i]]
        else:
            self.buffer['act'][pointer] = act
        self.buffer['reward'][pointer] = reward
        self.buffer['state_'][pointer] = state_
        self.buffer['done'][pointer] = done
        self.pointer += 1

    def get_training_data(self, index):
        for key in self.buffer.keys():
            k = 0
            for i in index:
                self.batch[key][k] = torch.tensor(self.buffer[key][i], dtype=torch.float32, device=self.device)
                k += 1
        return self.batch

    def store_buffer(self, agent):
        with open("./data/DDPG_{}.pkl".format(agent), "wb") as tf:
            pickle.dump(self.buffer, tf)

    def load_buffer(self, agent):
        with open("./data/DDPG_{}.pkl".format(agent), "rb") as tf:
            self.buffer = pickle.load(tf)
