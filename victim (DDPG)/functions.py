import numpy as np
import pandas as pd
from pettingzoo.mpe import simple_tag_v2, simple_world_comm_v2
from ddpg import DDPG

def make_env(scenario, args):
    if scenario == "predator_prey":
        env = simple_tag_v2.parallel_env(num_good=args.num_attacker, num_adversaries=args.num_victim,
                                         num_obstacles=2, max_cycles=args.episode_limit,
                                         continuous_actions=args.continuous_actions)
    else:  # scenario == "world_comm":
        env = simple_world_comm_v2.parallel_env(num_good=args.num_attacker, num_adversaries=args.num_victim,
                                                num_obstacles=2, num_food=args.num_attacker,
                                                max_cycles=args.episode_limit, num_forests=args.num_forests,
                                                continuous_actions=args.continuous_actions)
    return env

def env_information(env, args):
    obs_dim = [env.observation_space(env.agents[i]).shape[0] for i in range(env.num_agents)]
    if args.continuous_actions:
        act_dim = [env.action_space(env.agents[i]).shape[0] for i in range(env.num_agents)]
        act_bound = [env.action_space(env.agents[i]).high for i in range(env.num_agents)]
    else:
        act_dim = [env.action_space(env.agents[i]).n for i in range(env.num_agents)]
        act_bound = [0 for _ in range(env.num_agents)]
    return obs_dim, act_dim, act_bound

def get_agents(args):
    agents = []
    for i in range(args.num_victim):
        agent = DDPG(args, args.obs_dim[i], args.act_dim[i], args.act_bound[i])
        agents.append(agent)
    for i in range(args.num_victim, args.num_agents):
        agent = DDPG(args, args.obs_dim[i], args.act_dim[i], args.act_bound[i])
        agents.append(agent)

    return agents

def load_victim(args, agents, version):
    for index in range(args.num_victim):
        agents[index].load_model(args.load_dir, index, args.seed, version)

def load_attacker(args, agents, version):
    for index in range(args.num_victim, args.num_agents):
        agents[index].load_model(args.load_dir, index, args.seed, version)

def save_agents(args, agents, version):
    for index in range(args.num_agents):
        agents[index].save_model(args.save_dir, index, args.seed, version)

def save_rewards(args, reward_buffer, reward_vic, reward_att, num_cat, cat_rate):
    reward_buffer.to_csv('{}/reward_buffer_seed_{}.csv'.format(args.reward_dir, args.seed))
    reward_victim = np.array(reward_vic)
    reward_attacker = np.array(reward_att)
    num_catch = np.array(num_cat)
    catch_rate = np.array(cat_rate)
    np.save('{}/reward_victim_seed_{}.npy'.format(args.reward_dir, args.seed), reward_victim)
    np.save('{}/reward_attacker_seed_{}.npy'.format(args.reward_dir, args.seed), reward_attacker)
    np.save('{}/num_catch_seed_{}.npy'.format(args.reward_dir, args.seed), num_catch)
    np.save('{}/catch_rate_seed_{}.npy'.format(args.reward_dir, args.seed), catch_rate)

def load_rewards(args):
    reward_buffer = pd.read_csv('{}/reward_buffer_seed_{}.csv'.format(args.reward_dir, args.seed), index_col=0)
    reward_victim = np.load('{}/reward_victim_seed_{}.npy'.format(args.reward_dir, args.seed))
    reward_attacker = np.load('{}/reward_attacker_seed_{}.npy'.format(args.reward_dir, args.seed))

    return reward_buffer, reward_victim, reward_attacker

def reward_unified(reward_, env_agents, args):
    num_catch = 0
    judge_catch = 0
    rew_n = np.array(list(reward_.values()))
    rew_v = rew_n[:args.num_victim]
    rew_a = rew_n[args.num_victim:]
    rew_min = min(rew_a)
    for i in range(args.num_attacker):
        num_catch += abs(round(rew_a[i] / 100))
        rew_a[i] = rew_min
    rew_ = np.append(rew_v, rew_a)
    rew = {agent: j for agent, j in zip(env_agents, rew_)}
    if num_catch > 0:
        judge_catch = 1
    return rew, num_catch, judge_catch

def learn_judge(args, num_learn, agents, num_episode):
    num_learn_ = num_learn
    if (agents[0].buffer.pointer - 1) >= args.memory_capacity and \
            (agents[0].buffer.pointer - 1) % args.episode_limit == 0:
        num_learn += 1

        if args.task == "train_all":
            for agent in agents:
                agent.train()

        elif args.task == "train_victim":
            agent_learn = 0
            for agent in agents:
                if agent_learn < args.num_victim:
                    agent.train()
                agent_learn += 1

        elif args.task == "train_attacker":
            agent_learn = 0
            for agent in agents:
                if agent_learn >= args.num_victim:
                    agent.train()
                agent_learn += 1

    # learning rate decay
    if args.lr_decay and num_learn_ != num_learn:
        lr = args.lr * (1 - num_episode / args.max_episode)
    else:
        lr = args.lr

    return num_learn, lr
