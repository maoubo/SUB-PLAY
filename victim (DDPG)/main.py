import argparse
import datetime
from functions import *
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Runner_DDPG:
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

        # self.args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.args.device = torch.device("cpu")

        # init sub_policies
        self.agents = get_agents(self.args)
        if self.args.load_victim:
            load_victim(self.args, self.agents, 1)
        if self.args.load_attacker:
            load_attacker(self.args, self.agents, 1)

        self.reward_buffer = pd.DataFrame(
            np.ones((self.args.max_episode, self.args.num_agents)) * 0, columns=self.env_agents)
        self.reward_vic = []
        self.reward_att = []
        self.num_catch = []
        self.catch_rate = []
        self.num_episode = 0
        self.num_train = 0
        self.version = self.args.version

    def run(self, ):
        while self.num_episode < self.args.max_episode:
            ep_reward, _, _ = self.run_episode_mpe(evaluate=False)   # run an episode
            self.reward_buffer.iloc[self.num_episode] = ep_reward
            self.num_episode += 1

            if self.num_episode % self.args.save_freq == 0:
                self.version += 1

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

                save_rewards(self.args, self.reward_buffer, self.reward_vic,
                             self.reward_att, self.num_catch, self.catch_rate)
                save_agents(self.args, self.agents, self.version)

                # plt.plot(self.num_catch)
                # plt.plot(self.catch_rate)
                # plt.show()

        self.env.close()

    def evaluate(self, ):
        reward_vic = np.zeros(self.args.evaluate_episode)
        reward_att = np.zeros(self.args.evaluate_episode)
        num_catch = np.zeros(self.args.evaluate_episode)
        judge_catch = np.zeros(self.args.evaluate_episode)
        for i in range(self.args.evaluate_episode):
            ep_reward, ep_num_catch, ep_judge_catch = self.run_episode_mpe(evaluate=True)
            reward_vic[i] = sum(ep_reward[:self.args.num_victim]) / self.args.num_victim
            reward_att[i] = sum(ep_reward[self.args.num_victim:]) / self.args.num_attacker
            num_catch[i] = ep_num_catch
            judge_catch[i] = ep_judge_catch

        return reward_vic, reward_att, num_catch, judge_catch

    def run_episode_mpe(self, evaluate=False):
        ep_reward = {agent: 0.0 for agent in self.env_agents}
        ep_num_catch = 0
        ep_judge_catch = 0
        for episode_step in range(self.args.episode_limit):
            act_ = [self.agents[index].choose_action(list(self.obs_n.values())[index], self.args.act_bound[index])
                    for index in range(self.args.num_agents)]
            act_n = {agent: act_ for agent, act_ in zip(self.env_agents, act_)}
            obs_n_, rew_, done_n, info_n = self.env.step(act_n)

            rew_n, step_num_catch, step_judge_catch = reward_unified(rew_, self.env_agents, self.args)
            ep_num_catch += step_num_catch
            if step_judge_catch > 0:
                ep_judge_catch = 1

            if not evaluate:
                index = 0
                keys = list(self.obs_n.keys())
                for agent in self.agents:
                    agent.buffer.store_transition(self.obs_n[keys[index]], act_n[keys[index]], rew_n[keys[index]],
                                                  obs_n_[keys[index]], done_n[keys[index]])
                    index += 1

                self.num_train, self.args.lr = learn_judge(self.args, self.num_train, self.agents, self.num_episode)

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

        return ep_reward, ep_num_catch, ep_judge_catch

    def test(self,):
        if self.args.load_victim:
            load_victim(self.args, self.agents, self.version)
        if self.args.load_attacker:
            load_attacker(self.args, self.agents, self.version)
        rew_vic = np.zeros(self.args.test_episode)
        rew_att = np.zeros(self.args.test_episode)
        num_cat = np.zeros(self.args.test_episode)
        catch_rate = np.zeros(self.args.test_episode)

        for i in range(self.args.test_episode):
            reward_vic, reward_att, num_catch, judge_catch = self.evaluate()
            rew_vic[i] = reward_vic.mean()
            rew_att[i] = reward_att.mean()
            num_cat[i] = num_catch.mean()
            catch_rate[i] = judge_catch.mean()

        print("Victim Reward : {:.4f} | Attacker Reward : {:.4f} | Num_catch : {:.4f}"
              "| Catch Rate : {:.4f}".format(rew_vic.mean(), rew_att.mean(), num_cat.mean(), catch_rate.mean()))

        self.env.close()

        return num_cat.mean(), catch_rate.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG in MPE environment")
    # Environment
    parser.add_argument("--scenario", type=str, default="predator_prey", help="predator_prey / world_comm")
    parser.add_argument("--continuous_actions", type=bool, default=True, help="type of action")
    parser.add_argument("--episode_limit", type=int, default=25, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=100000, help="maximum episode")
    parser.add_argument("--evaluate_episode", type=int, default=100, help="number of episodes for testing")
    parser.add_argument("--test_episode", type=int, default=10, help="number of episodes for testing")
    parser.add_argument("--num_agents", type=int, default=4, help="number of agents")
    parser.add_argument("--num_victim", type=int, default=3, help="number of predators")
    parser.add_argument("--num_attacker", type=int, default=1, help="number of preys")
    parser.add_argument("--num_forests", type=int, default=1, help="number of Forests in world_comm")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="save the model every 'save_freq' episodes")
    parser.add_argument("--save_dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--load_dir", type=str, default="./model",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--reward_dir", type=str, default="./reward", help="directory where rewards are saved")
    parser.add_argument("--version", type=int, default=0, help="version")
    # Hyper-parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="soft replacement")
    parser.add_argument("--var", type=float, default=0.01, help="action noise")
    parser.add_argument("--memory_capacity", type=int, default=200000, help="size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize concurrently")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="adam epsilon")
    # Important options
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--task", type=str, default="train_all", help="train_all / train_victim / train_attacker")
    parser.add_argument("--load_victim", type=bool, default=False, help="whether to load victim")
    parser.add_argument("--load_attacker", type=bool, default=False, help="whether to load attacker")
    parser.add_argument("--render", type=bool, default=False, help="visualization or not")
    parser.add_argument("--lr_decay", type=bool, default=False, help="learning rate decay or not")

    args = parser.parse_args()
    # runner = Runner_DDPG(args)
    # runner.run()
    # runner.test()

    # -------------train----------------
    for seed_ in range(3):
        args = parser.parse_args()
        args.seed = seed_
        args.render = False
        args.lr_decay = False
        args.load_victim = False
        args.load_attacker = False
        args.save_dir = "./result/seed{}/model".format(args.seed)
        args.load_dir = "./result/seed{}/model".format(args.seed)
        args.reward_dir = "./result/seed{}/reward".format(args.seed)
        runner = Runner_DDPG(args)
        runner.run()

    # -------------test----------------
    for seed_ in range(3):
        test_record = pd.DataFrame(np.ones((100, 2)) * 0, columns=["num_cat", "catch_rate"])
        version = 1
        while version <= 100:
            print("------{}-------".format(version))
            args = parser.parse_args()
            args.seed = seed_
            args.save_dir = "./result/seed{}/model".format(args.seed)
            args.load_dir = "./result/seed{}/model".format(args.seed)
            args.reward_dir = "./result/seed{}/reward".format(args.seed)
            args.version = version
            args.load_victim = True
            args.load_attacker = False
            runner = Runner_DDPG(args)
            test_record.iloc[version - 1] = runner.test()
            version += 1
            test_record.to_csv('{}/test_record_seed_{}.csv'.format(args.reward_dir, seed_))
