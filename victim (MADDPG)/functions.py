import numpy as np
import pandas as pd
from pettingzoo.mpe import simple_tag_v2, simple_world_comm_v2
from maddpg import MADDPG

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
        agent = MADDPG(args, args.obs_dim, args.act_dim, args.act_bound, i)
        agents.append(agent)
    for i in range(args.num_victim, args.num_agents):
        agent = MADDPG(args, args.obs_dim, args.act_dim, args.act_bound, i)
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

def format_conversion(args, obs_n, act_n, rew_n, obs_n_, done_n):

    obs_vic = np.array(list(obs_n.values())[0:args.num_victim])
    if args.scenario == 'world_comm':
        index = 0
        for key in act_n.keys():
            act_n[key] = np.pad(act_n[key], (0, args.act_dim[0] - args.act_dim[index]), 'constant')
            index += 1
    act_vic = np.array(list(act_n.values())[0:args.num_victim])
    rew_vic = np.array(list(rew_n.values())[0:args.num_victim]).reshape(args.num_victim, 1)
    obs_vic_ = np.array(list(obs_n_.values())[0:args.num_victim])
    done_vic = np.array(list(done_n.values())[0:args.num_victim]).reshape(args.num_victim, 1)

    obs_att = np.array(list(obs_n.values())[args.num_victim:])
    act_att = np.array(list(act_n.values())[args.num_victim:])
    rew_att = np.array(list(rew_n.values())[args.num_victim:]).reshape(args.num_attacker, 1)
    obs_att_ = np.array(list(obs_n_.values())[args.num_victim:])
    done_att = np.array(list(done_n.values())[args.num_victim:]).reshape(args.num_attacker, 1)

    return obs_vic, act_vic, rew_vic, obs_vic_, done_vic, obs_att, act_att, rew_att, obs_att_, done_att

def learn_judge(args, buffer_vic, buffer_att, num_learn, agents, num_episode):
    num_learn_ = num_learn
    if (buffer_vic.pointer - 1) >= args.memory_capacity and (buffer_vic.pointer - 1) % args.episode_limit == 0:
        num_learn += 1

        if args.task == "train_all":
            agent_id = 0
            for agent in agents:
                if agent_id < args.num_victim:
                    agent.train(agents, buffer_vic, agent_id, 0)
                elif agent_id >= args.num_victim:
                    agent.train(agents, buffer_att, agent_id, 1)
                agent_id += 1

        elif args.task == "train_victim":
            agent_id = 0
            for agent in agents:
                if agent_id < args.num_victim:
                    agent.train(agents, buffer_vic, agent_id, 0)
                agent_id += 1

        elif args.task == "train_attacker":
            agent_id = 0
            for agent in agents:
                if agent_id >= args.num_victim:
                    agent.train(agents, buffer_att, agent_id, 1)
                agent_id += 1

    # learning rate decay
    if args.lr_decay and num_learn_ != num_learn:
        lr = args.lr * (1 - num_episode / args.max_episode)
    else:
        lr = args.lr

    return num_learn, lr


