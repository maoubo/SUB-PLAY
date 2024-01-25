import numpy as np
import torch
import pickle

class ReplayBuffer:
    def __init__(self, args, obs_dim, act_dim):
        self.memory_capacity = args.memory_capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = args.device
        self.pointer = 0
        self.buffer = {'state': np.zeros([self.memory_capacity, self.obs_dim]),
                       'act': np.zeros([self.memory_capacity, self.act_dim]),
                       'reward': np.zeros([self.memory_capacity, 1]),
                       'state_': np.zeros([self.memory_capacity, self.obs_dim]),
                       'done': np.zeros([self.memory_capacity, 1])}
        self.batch = {'state': torch.zeros([args.batch_size, self.obs_dim]),
                      'act': torch.zeros([args.batch_size, self.act_dim]),
                      'reward': torch.zeros([args.batch_size, 1]),
                      'state_': torch.zeros([args.batch_size, self.obs_dim]),
                      'done': torch.zeros([args.batch_size, 1])}

    def store_transition(self, state, act, reward, state_, done):
        pointer = self.pointer % self.memory_capacity
        self.buffer['state'][pointer] = state
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
