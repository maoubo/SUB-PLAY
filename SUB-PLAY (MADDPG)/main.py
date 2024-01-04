import argparse
import datetime
from functions import *
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from replay_buffer import ReplayBuffer

class Runner_MADDPG:
    def __init__(self, args):
        self.args = args

        # init env
        self.env = make_env(args.scenario, args)
        self.obs_n = self.env.reset()
        self.env_agents = self.env.agents

        # set random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.args.obs_dim, self.args.act_dim, self.args.act_bound = env_information(self.env, args)

        self.args.obs_dim_vic = [self.args.obs_dim[i] for i in range(self.args.num_victim)]
        self.args.obs_dim_att = [self.args.obs_dim[i] for i in range(self.args.num_victim, self.args.num_agents)]
        self.args.act_dim_vic = [self.args.act_dim[i] for i in range(self.args.num_victim)]
        self.args.act_dim_att = [self.args.act_dim[i] for i in range(self.args.num_victim, self.args.num_agents)]

        # self.args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.args.device = torch.device("cpu")

        self.agents = []
        self.buffer_vic = []
        self.buffer_att = []

        # init sub_policies
        for sub in range(self.args.num_victim + 1):
            agents = get_agents(self.args)
            if self.args.load_victim:
                load_victim(self.args, agents)
            if self.args.load_attacker:
                load_attacker(self.args, agents)
            self.agents.append(agents)

            # init replay_buffer
            self.buffer_vic.append(ReplayBuffer(self.args, self.args.obs_dim_vic, self.args.act_dim_vic, 0))
            self.buffer_att.append(ReplayBuffer(self.args, self.args.obs_dim_att, self.args.act_dim_att, 1))

        self.reward_buffer = pd.DataFrame(
            np.ones((self.args.max_episode, self.args.num_agents)) * 0, columns=self.env_agents)
        self.reward_vic = []
        self.reward_att = []
        self.num_catch = []
        self.catch_rate = []
        self.num_episode = 0
        self.num_train = 0
        self.min_num_catch = 50
        self.min_catch_rate = 1

        # init experience dissemination table
        self.store_rate, self.tran_rate = store_tran_rate(self.args, self.args.mask_rate, [], False)
        self.sum_sub = []

    def run(self, ):
        if self.args.mask_type == "distance" or self.args.mask_type == "region":
            self.stat(update=False)

        test_record = pd.DataFrame(index=range(50), columns=["PM"])
        index_ = 0

        while self.num_episode < self.args.max_episode:
            ep_reward, _, _, self.sum_sub = self.run_episode_mpe(evaluate=False)  # run an episode
            self.reward_buffer.iloc[self.num_episode] = ep_reward
            self.num_episode += 1

            # dynamic observation
            if self.args.mask_type == "distance" or self.args.mask_type == "region":
                self.stat(update=True)

            if self.num_episode % self.args.save_freq == 0:
                # test
                reward_vic, reward_att, num_catch, judge_catch = self.evaluate()
                self.reward_vic.append(reward_vic.mean())
                self.reward_att.append(reward_att.mean())
                self.num_catch.append(num_catch.mean())
                self.catch_rate.append(judge_catch.mean())
                print("{} | Episode: {}/{} | Num_train: {} | Victim Reward: {:.4f} | Attacker Reward: {:.4f} | "
                      " Num_catch: {:.4f} | Catch Rate: {:.4f}".
                      format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.num_episode,
                             self.args.max_episode, self.num_train,
                             reward_vic.mean(), reward_att.mean(), num_catch.mean(), judge_catch.mean()))

                # policy meritocracy
                pm = (2 * num_catch.mean() * judge_catch.mean()) / (num_catch.mean() + judge_catch.mean())
                pm_ = (2 * self.min_num_catch * self.min_catch_rate) / (self.min_num_catch + self.min_catch_rate)

                if pm <= pm_:
                    self.min_num_catch = num_catch.mean()
                    self.min_catch_rate = judge_catch.mean()
                    test_record.iloc[index_] = pm
                    save_agents(self.args, self.agents)
                    np.save('{}/store_rate_{}.npy'.format(self.args.reward_dir, self.args.seed), self.store_rate)

                else:
                    test_record.iloc[index_] = pm_

                index_ += 1
                save_rewards(self.args, self.reward_buffer, self.reward_vic,
                             self.reward_att, self.num_catch, self.catch_rate)

                # plt.plot(self.reward_vic)
                # plt.plot(self.reward_att)
                # plt.show()

        test_record.to_csv('{}/test_record_seed_{}.csv'.format(args.reward_dir, self.args.seed))
        self.env.close()

    def stat(self, update=False):
        if not update:
            # observing the observable probability of the victim before attack
            sum_sub = np.zeros(self.args.num_victim + 1)

            for _ in range(self.args.num_stat):
                _, _, _, sum_sub_ = self.run_episode_mpe(evaluate=True)
                sum_sub += sum_sub_

            sum_sub = sum_sub / (self.args.num_stat * self.args.episode_limit * self.args.num_attacker)

        else:
            # dynamically updating the experience dissemination table
            sum_sub = (self.args.num_stat / (self.args.num_stat + 1)) * self.store_rate \
                      + (1 / (self.args.num_stat + 1)) * \
                      self.sum_sub / (self.args.episode_limit * self.args.num_attacker)

        sub = np.linspace(0, self.args.num_victim, self.args.num_victim + 1)
        self.args.mask_rate = 1 - (sum(sum_sub * sub) / self.args.num_victim)
        self.store_rate, self.tran_rate = store_tran_rate(self.args, self.args.mask_rate, sum_sub, True)

    def evaluate(self, ):
        reward_vic = np.zeros(self.args.evaluate_episode)
        reward_att = np.zeros(self.args.evaluate_episode)
        num_catch = np.zeros(self.args.evaluate_episode)
        judge_catch = np.zeros(self.args.evaluate_episode)
        for num_eval in range(self.args.evaluate_episode):
            ep_reward, ep_num_catch, ep_judge_catch, _ = self.run_episode_mpe(evaluate=True)
            reward_vic[num_eval] = sum(ep_reward[:self.args.num_victim]) / self.args.num_victim
            reward_att[num_eval] = sum(ep_reward[self.args.num_victim:]) / self.args.num_attacker
            num_catch[num_eval] = ep_num_catch
            judge_catch[num_eval] = ep_judge_catch

        return reward_vic, reward_att, num_catch, judge_catch

    def run_episode_mpe(self, evaluate=False):
        sum_sub = np.zeros(self.args.num_victim + 1)
        ep_reward = {agent: 0.0 for agent in self.env_agents}
        ep_num_catch = 0
        ep_judge_catch = 0
        mask, obs_adv = partial_mask(self.args, self.env_agents, self.obs_n)
        self.obs_n = partial_observation(self.args, self.env_agents, self.obs_n, mask)

        for episode_step in range(self.args.episode_limit):
            # sub_policy ensemble
            act_ = policy_ensemble(self.args, self.agents, self.obs_n, obs_adv, self.store_rate)
            act_n = {agent: act_ for agent, act_ in zip(self.env_agents, act_)}

            obs_n_, rew_, done_n, info_n = self.env.step(act_n)

            # stat the number of occurrences of all sub games.
            sum_sub = stat_sub(args, sum_sub, obs_adv)

            # the mask for each step of distance-based and region-based may change, while uncertainty-based is fixed.
            if self.args.mask_type == 'distance' or self.args.mask_type == 'region':
                mask, obs_adv = partial_mask(self.args, self.env_agents, obs_n_)
            obs_n_ = partial_observation(self.args, self.env_agents, obs_n_, mask)
            rew_n, step_num_catch, step_judge_catch = reward_unified(rew_, self.env_agents, self.args)
            ep_num_catch += step_num_catch

            if step_judge_catch > 0:
                ep_judge_catch = 1

            if not evaluate:
                # record transitions
                obs_vic, act_vic, rew_vic, obs_vic_, done_vic, obs_att, act_att, rew_att, obs_att_, done_att \
                    = format_conversion(self.args, self.obs_n, act_n, rew_n, obs_n_, done_n)

                for sub in range(self.args.num_victim + 1):
                    agents = self.agents[sub]
                    for agent_id in range(self.args.num_agents):
                        if agent_id >= self.args.num_victim:
                            tran_rate = self.tran_rate[int(sum(obs_adv[agent_id - self.args.num_victim])), sub]
                            tran_judge = np.random.choice([0, 1], size=1, p=[1 - tran_rate, tran_rate])

                            if tran_judge == 1:
                                self.buffer_vic[sub].store_transition(obs_vic, act_vic, rew_vic, obs_vic_, done_vic)
                                self.buffer_att[sub].store_transition(obs_att, act_att, rew_att, obs_att_, done_att)

                    # determine which agents need to be updated
                    self.num_train = learn_judge(
                        self.args, self.buffer_vic[sub], self.buffer_att[sub], self.num_train, agents)

            # terminal state judgment
            judge = False
            for agent in self.env_agents:
                if done_n[agent]:
                    judge = True

            if judge or episode_step == self.args.episode_limit - 1:
                self.obs_n = self.env.reset()
                ep_reward = list(ep_reward.values())

            else:
                self.obs_n = obs_n_
                ep_reward = {agent: ep_reward[agent] + rew_n[agent] for agent in self.env_agents}

            if self.args.render:
                time.sleep(0.1)
                self.env.render()

        return ep_reward, ep_num_catch, ep_judge_catch, sum_sub

    def test(self, ):
        for sub in range(self.args.num_victim + 1):
            if self.args.load_victim:
                load_victim(self.args, self.agents[sub])
            if self.args.load_attacker:
                load_sub_attacker(self.args, self.agents[sub], sub)

        if self.args.mask_type == "distance" or self.args.mask_type == "region":
            self.store_rate = np.load('{}/store_rate_{}.npy'.format(self.args.reward_dir, self.args.seed))

        rew_vic = np.zeros(self.args.test_episode)
        rew_att = np.zeros(self.args.test_episode)
        num_cat = np.zeros(self.args.test_episode)
        catch_rate = np.zeros(self.args.test_episode)

        for num_test in range(self.args.test_episode):
            reward_vic, reward_att, num_catch, judge_catch = self.evaluate()
            rew_vic[num_test] = reward_vic.mean()
            rew_att[num_test] = reward_att.mean()
            num_cat[num_test] = num_catch.mean()
            catch_rate[num_test] = judge_catch.mean()

        print("Victim Reward : {:.4f} | Attacker Reward : {:.4f} | Num_catch : {:.4f}"
              "| Catch Rate : {:.4f}".format(rew_vic.mean(), rew_att.mean(), num_cat.mean(), catch_rate.mean()))

        self.env.close()

        return rew_vic.mean(), rew_att.mean(), num_cat.mean(), catch_rate.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG in MPE Environment")
    # Environment
    parser.add_argument("--scenario", type=str, default="predator_prey", help="predator_prey / world_comm")
    parser.add_argument("--continuous_actions", type=bool, default=True, help="type of action")
    parser.add_argument("--episode_limit", type=int, default=25, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=3000, help="maximum episode")
    parser.add_argument("--evaluate_episode", type=int, default=100, help="number of episodes for testing")
    parser.add_argument("--test_episode", type=int, default=10, help="number of episodes for testing")
    parser.add_argument("--num_stat", type=int, default=10, help="number of episodes for statistics")
    parser.add_argument("--num_agents", type=int, default=5, help="number of agents in the env")
    parser.add_argument("--num_victim", type=int, default=3, help="number of predators")
    parser.add_argument("--num_attacker", type=int, default=2, help="number of preys")
    parser.add_argument("--num_forests", type=int, default=1, help="number of forests in world_comm")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=100,
                        help="save the model every 'save_freq' episodes")
    parser.add_argument("--save_dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--load_dir", type=str, default="./model",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--reward_dir", type=str, default="./reward", help="directory where rewards are saved")
    # Hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="soft replacement")
    parser.add_argument("--var", type=float, default=0.01, help="action noise")
    parser.add_argument("--memory_capacity", type=int, default=512, help="size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=512, help="number of episodes to optimize concurrently")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="adam epsilon")
    # Important options
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--task", type=str, default="train_attacker", help="train_all / train_victim / train_attacker")
    parser.add_argument("--load_victim", type=bool, default=True, help="whether to load victim")
    parser.add_argument("--load_attacker", type=bool, default=False, help="whether to load attacker")
    parser.add_argument("--render", type=bool, default=False, help="visualization or not")
    parser.add_argument("--mask_type", type=str, default="random", help="random / distance / region")
    parser.add_argument("--mask_rate", type=float, default=0.0, help="probability that the target cannot be observed")
    parser.add_argument("--distance_range", type=float, default=1.0, help="visible distance")

    args = parser.parse_args()
    # runner = Runner_MADDPG(args)
    # runner.run()
    # runner.test()

    """
    uncertainty-based
    """
    # -------------train----------------
    args.mask_type = "random"
    args.load_victim = True
    args.load_attacker = False
    mask_rate = "0.25"
    for seed in range(3):
        args.seed = seed
        print("--------seed: {} mask_rate: {}-----------".format(args.seed, mask_rate))
        args.mask_rate = float(mask_rate)

        args.save_dir = "./data/seed{}/model".format(args.seed)
        args.load_dir = "./data/seed{}/model".format(args.seed)
        args.reward_dir = "./data/seed{}/reward".format(args.seed)

        runner = Runner_MADDPG(args)
        runner.run()

    # -------------test----------------
    args.mask_type = "random"
    args.load_victim = True
    args.load_attacker = True
    result = pd.DataFrame(index=range(144), columns=["Victim Reward", "Attacker Reward", "Num_catch", "Catch Rate"])
    index = 0
    mask_rate = "0.25"
    for seed in range(3):
        args.seed = seed
        args.save_dir = "./data/seed{}/model".format(args.seed)
        args.load_dir = "./data/seed{}/model".format(args.seed)
        args.reward_dir = "./data/seed{}/reward".format(args.seed)

        print("--------seed: {} train: {} test: {}-----------".format(args.seed, mask_rate, mask_rate))
        args.mask_rate = float(mask_rate)
        runner = Runner_MADDPG(args)
        a1, b1, c1, d1 = runner.test()
        result.iloc[index, :] = [a1, b1, c1, d1]
        index += 1
        result.to_csv('./result.csv')

