import numpy as np
from network import *
from ema import EMA
from replay_buffer import ReplayBuffer

class DDPG(object):
    def __init__(self, args, obs_dim, act_dim, act_bound):
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound

        self.actor_input_dim = obs_dim
        self.actor_output_dim = act_dim
        self.critic_input_dim = obs_dim + act_dim

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

        # create replay_buffer
        self.buffer = ReplayBuffer(self.args, self.obs_dim, self.act_dim)

    def choose_action(self, obs, act_bound, train=True):
        obs = torch.tensor(obs)
        obs = obs.to(self.args.device)
        act = self.actor(obs).detach().cpu().numpy()
        act = self.act_bound * act
        if train:
            # add noise for exploration
            act = np.clip(np.random.normal(act, self.args.var), -act_bound, act_bound)
        return act

    def get_critic_loss(self, batch):

        state = batch['state']
        act = batch['act']
        reward = batch['reward']
        state_ = batch['state_']
        state = state.to(self.args.device)
        act = act.to(self.args.device)
        reward = reward.to(self.args.device)
        state_ = state_.to(self.args.device)

        act_ = self.actor_target(state_)
        q_ = self.critic_target(torch.cat([state_, act_], dim=1))
        q_target = reward + self.args.gamma * q_
        q = self.critic(torch.cat([state, act], dim=1))
        critic_loss = self.loss_td(q_target, q)

        return critic_loss

    def get_actor_loss(self, batch):

        state = batch['state']
        state = state.to(self.args.device)
        act = self.actor(state)
        q = self.critic(torch.cat([state, act], dim=1))
        actor_loss = -q.mean()

        return actor_loss

    def train(self):
        index = np.random.choice(self.args.memory_capacity, size=self.args.batch_size, replace=False, p=None)
        batch = self.buffer.get_training_data(index)

        self.critic_optimizer.zero_grad()
        critic_loss = self.get_critic_loss(batch)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = self.get_actor_loss(batch)
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
