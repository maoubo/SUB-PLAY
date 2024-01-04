import torch
import torch.nn as nn

def param_init(layer):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.zeros_(param)
        elif 'weight' in name:
            nn.init.xavier_normal_(param, gain=1.0)

class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim, actor_output_dim, act_bound):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, actor_output_dim)
        self.activate_ReLU = nn.ReLU()
        self.activate_Tanh = nn.Tanh()
        param_init(self.fc1)
        param_init(self.fc2)
        param_init(self.fc3)
        self.act_bound = torch.tensor(act_bound)

    def forward(self, actor_input):
        fc1 = self.fc1(actor_input)
        fc1 = self.activate_ReLU(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.activate_ReLU(fc2)
        fc3 = self.fc3(fc2)
        fc3 = self.activate_Tanh(fc3)
        return fc3

class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_ReLU = nn.ReLU()
        param_init(self.fc1)
        param_init(self.fc2)
        param_init(self.fc3)

    def forward(self, critic_input):
        fc1 = self.fc1(critic_input)
        fc1 = self.activate_ReLU(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.activate_ReLU(fc2)
        value = self.fc3(fc2)
        return value
