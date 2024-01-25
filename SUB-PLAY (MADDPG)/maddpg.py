import numpy as np
from network import *
from ema import EMA

class MADDPG(object):
    def __init__(self, args, obs_dim, act_dim, act_bound, agent_id):
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound[agent_id]
        self.agent_id = agent_id

        self.actor_input_dim = obs_dim[agent_id]
        self.actor_output_dim = act_dim[agent_id]

        if agent_id < args.num_victim:
            self.critic_input_dim = sum(obs_dim[0:args.num_victim]) + sum(act_dim[0:args.num_victim])
        else:
            self.critic_input_dim = sum(obs_dim[args.num_victim:]) + sum(act_dim[args.num_victim:])

        self.actor = Actor_MLP(self.args, self.actor_input_dim, self.actor_output_dim, self.act_bound)
        self.actor.to(self.args.device)
        self.actor_target = Actor_MLP(self.args, self.actor_input_dim, self.actor_output_dim, self.act_bound)
        self.actor_target.to(self.args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.train()
        self.actor_target.eval()

        self.critic = Critic_MLP(self.args, self.critic_input_dim)
        self.critic.to(self.args.device)
        self.critic_target = Critic_MLP(self.args, self.critic_input_dim)
        self.critic_target.to(self.args.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.train()
        self.critic_target.eval()

        self.loss_td = nn.MSELoss(reduction='mean')
        self.loss_td.to(self.args.device)

        # soft replacement
        self.ema_a = EMA(self.actor_target, 1 - self.args.tau)
        self.ema_a.register()
        self.ema_c = EMA(self.critic_target, 1 - self.args.tau)
        self.ema_c.register()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr, eps=self.args.adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr, eps=self.args.adam_eps)

    def choose_action(self, obs, act_bound, train=True):
        obs = torch.tensor(obs)
        obs = obs.to(self.args.device)
        act = self.actor(obs).detach().cpu().numpy()
        act = self.act_bound * act
        if train:
            # add noise for exploration
            act = np.clip(np.random.normal(act, self.args.var), -act_bound, act_bound)
        return act

    def get_critic_loss(self, agents, batch, agent_id, buffer_type):
        state = batch['state']
        act = batch['act']
        reward = batch['reward']
        state_ = batch['state_']
        state = state.to(self.args.device)
        act = act.to(self.args.device)
        reward = reward.to(self.args.device)
        state_ = state_.to(self.args.device)

        if buffer_type == 0:
            act_n = []
            for i in range(self.args.num_victim):
                act_i = torch.zeros(self.args.batch_size, 1, self.act_dim[i])
                for j in range(self.args.batch_size):
                    act_i[j] = act[j][i][:self.act_dim[i]]
                act_n.append(act_i)
            act = torch.cat(act_n, dim=-1)
            act_ = self.get_act_(self.args.num_victim, agents, agent_id, state_, buffer_type)

        else:
            act_n = torch.zeros(self.args.batch_size, self.args.num_attacker, self.act_dim[agent_id])
            for i in range(self.args.batch_size):
                act_n[i] = act[i]
            act_ = self.get_act_(self.args.num_attacker, agents, agent_id, state_, buffer_type)

        state = state.reshape(self.args.batch_size, -1)
        act = act.reshape(self.args.batch_size, -1)
        state_ = state_.reshape(self.args.batch_size, -1)
        act_ = act_.reshape(self.args.batch_size, -1)

        reward_id = torch.zeros(self.args.batch_size, 1)
        for i in range(self.args.batch_size):
            reward_id[i] = reward[i][agent_id - buffer_type * self.args.num_victim]

        critic_input_target = torch.cat([state_, act_], dim=1).reshape(self.args.batch_size, -1)
        critic_input = torch.cat([state, act], dim=1).reshape(self.args.batch_size, -1)

        with torch.no_grad():
            q_ = self.critic_target(critic_input_target)
            q_target = reward_id + self.args.gamma * q_
        q = self.critic(critic_input)
        critic_loss = self.loss_td(q_target, q)

        return critic_loss

    def get_act_(self, num, agents, agent_id, state_, buffer_type):
        act_ = []
        for i in range(num):
            state_agent = torch.zeros(self.args.batch_size, self.obs_dim[agent_id])
            if buffer_type == 0:
                for j in range(self.args.batch_size):
                    state_agent[j] = state_[j][agent_id]
                act_.append(agents[i].actor_target(state_agent))
            else:
                for j in range(self.args.batch_size):
                    state_agent[j] = state_[j][agent_id - buffer_type * self.args.num_victim]
                act_.append(agents[i + self.args.num_victim].actor_target(state_agent))
        act_ = torch.cat(act_, dim=-1)

        return act_

    def get_actor_loss(self, batch, agent_id, buffer_type):
        state = batch['state']
        act = batch['act']
        state = state.to(self.args.device)
        act = act.to(self.args.device)

        state_id = torch.zeros(self.args.batch_size, self.obs_dim[agent_id])
        for i in range(self.args.batch_size):
            state_id[i] = state[i][agent_id - buffer_type * self.args.num_victim]
        act_id = self.actor(state_id)

        if buffer_type == 0:
            act_ = []
            for i in range(self.args.num_victim):
                act_i = torch.zeros(self.args.batch_size, 1, self.act_dim[i])
                if i == agent_id:
                    act_i = act_id
                else:
                    for j in range(self.args.batch_size):
                        act_i[j] = act[j][i][:self.act_dim[i]]
                act_.append(act_i.reshape(self.args.batch_size, self.act_dim[i]))
            act_ = torch.cat(act_, dim=-1)

        else:
            act_ = torch.zeros(self.args.batch_size, self.args.num_attacker, self.act_dim[agent_id])
            for i in range(self.args.batch_size):
                act_[i] = act[i]
                act_[i][agent_id - buffer_type * self.args.num_victim] = act_id[i]

        state = state.reshape(self.args.batch_size, -1)
        act_ = act_.reshape(self.args.batch_size, -1)
        critic_input = torch.cat([state, act_], dim=1).reshape(self.args.batch_size, -1)
        q = self.critic(critic_input)
        actor_loss = -q.mean()

        return actor_loss

    def train(self, agents, buffer, agent_id, buffer_type):
        index = np.random.choice(self.args.memory_capacity, size=self.args.batch_size, replace=False, p=None)
        batch = buffer.get_training_data(index)

        self.critic_optimizer.zero_grad()
        critic_loss = self.get_critic_loss(agents, batch, agent_id, buffer_type)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = self.get_actor_loss(batch, agent_id, buffer_type)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.ema_a.update(self.actor)
        self.ema_a.apply_shadow()
        self.ema_c.update(self.critic)
        self.ema_c.apply_shadow()

    def save_model(self, save_dir, agent, seed):
        torch.save(self.actor.state_dict(), "{}/actor_{}_seed_{}.pth"
                   .format(save_dir, agent, seed))
        torch.save(self.actor_target.state_dict(), "{}/actor_target_{}_seed_{}.pth"
                   .format(save_dir, agent, seed))
        torch.save(self.critic.state_dict(), "{}/critic_{}_seed_{}.pth"
                   .format(save_dir, agent, seed))
        torch.save(self.critic_target.state_dict(), "{}/critic_target_{}_seed_{}.pth"
                   .format(save_dir, agent, seed))

    def load_model(self, load_dir, agent, seed):
        self.actor.load_state_dict(
            torch.load("{}/actor_{}_seed_{}.pth".format(load_dir, agent, seed)))
        self.actor_target.load_state_dict(
            torch.load("{}/actor_target_{}_seed_{}.pth".format(load_dir, agent, seed)))
        self.critic.load_state_dict(
            torch.load("{}/critic_{}_seed_{}.pth".format(load_dir, agent, seed)))
        self.critic_target.load_state_dict(
            torch.load("{}/critic_target_{}_seed_{}.pth".format(load_dir, agent, seed)))

    def save_sub_model(self, save_dir, agent, seed, sub):
        torch.save(self.actor.state_dict(), "{}/sub_model_{}/actor_{}_seed_{}_sub_{}.pth"
                   .format(save_dir, sub, agent, seed, sub))
        torch.save(self.actor_target.state_dict(), "{}/sub_model_{}/actor_target_{}_seed_{}_sub_{}.pth"
                   .format(save_dir, sub, agent, seed, sub))
        torch.save(self.critic.state_dict(), "{}/sub_model_{}/critic_{}_seed_{}_sub_{}.pth"
                   .format(save_dir, sub, agent, seed, sub))
        torch.save(self.critic_target.state_dict(), "{}/sub_model_{}/critic_target_{}_seed_{}_sub_{}.pth"
                   .format(save_dir, sub, agent, seed, sub))

    def load_sub_model(self, load_dir, agent, seed, sub):
        self.actor.load_state_dict(
            torch.load("{}/sub_model_{}/actor_{}_seed_{}_sub_{}.pth".format(load_dir, sub, agent, seed, sub)))
        self.actor_target.load_state_dict(
            torch.load("{}/sub_model_{}/actor_target_{}_seed_{}_sub_{}.pth".format(load_dir, sub, agent, seed, sub)))
        self.critic.load_state_dict(
            torch.load("{}/sub_model_{}/critic_{}_seed_{}_sub_{}.pth".format(load_dir, sub, agent, seed, sub)))
        self.critic_target.load_state_dict(
            torch.load("{}/sub_model_{}/critic_target_{}_seed_{}_sub_{}.pth".format(load_dir, sub, agent, seed, sub)))