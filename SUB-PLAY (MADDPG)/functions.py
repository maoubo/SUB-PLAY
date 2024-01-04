import numpy as np
import pandas as pd
import math
from pettingzoo.mpe import simple_tag_v2, simple_world_comm_v2
from maddpg import MADDPG
from scipy.special import comb

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

def store_tran_rate(args, mask_rate, store_rate_, replace):
    if not replace:  # args.mask_type == "random"
        store_rate = np.zeros(args.num_victim + 1)
        for sub in range(args.num_victim + 1):
            store_rate[sub] = comb(args.num_victim, sub) * math.pow(1 - mask_rate, sub) \
                                   * math.pow(mask_rate, args.num_victim - sub)

    else:  # args.mask_type == "distance" or args.mask_type == "region"
        store_rate = store_rate_

    h = 0.5
    tran_rate = np.zeros((args.num_victim + 1, args.num_victim + 1))
    for tran_type in range(args.num_victim + 1):
        for sub in range(args.num_victim + 1):
            if tran_type == sub:
                tran_rate[tran_type, sub] = 1
            else:
                if store_rate[sub] == 0 or store_rate[sub] == 1:
                    tran_rate[tran_type, sub] = 0
                else:
                    k = h - store_rate[sub]
                    if k < 0:
                        k = 0

                    tran_rate[tran_type, sub] = pow((k + np.std(store_rate)),
                                                    abs(tran_type - sub) / math.log(args.num_victim + 1))
                    tran_rate[tran_type, sub] = np.clip(tran_rate[tran_type, sub], 0, 1)

    return store_rate, tran_rate

def get_agents(args):
    agents = []
    for i in range(args.num_victim):
        agent = MADDPG(args, args.obs_dim, args.act_dim, args.act_bound, i)
        agents.append(agent)
    for i in range(args.num_victim, args.num_agents):
        agent = MADDPG(args, args.obs_dim, args.act_dim, args.act_bound, i)
        agents.append(agent)

    return agents

def load_victim(args, agents):
    for index in range(args.num_victim):
        agents[index].load_model(args.load_dir, index, args.seed)

def load_attacker(args, agents):
    for index in range(args.num_victim, args.num_agents):
        agents[index].load_model(args.load_dir, index, args.seed)

def load_sub_attacker(args, agents, sub):
    for index in range(args.num_victim, args.num_agents):
        agents[index].load_sub_model(args.load_dir, index, args.seed, sub)

def save_agents(args, agents):
    for sub in range(args.num_victim + 1):
        sub_agents = agents[sub]
        for index in range(args.num_agents):
            sub_agents[index].save_sub_model(args.save_dir, index, args.seed, sub)

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

def partial_mask(args, env_agents, obs_n):
    obs_adv = []  # obs_adv can be used for Mask generation and subsequent judgment in sub games
    mask = []

    # -----------testing sub-policy performance---------------
    # for _ in range(args.num_attacker):
    #
    #     if args.test_sub == 0:
    #         obs_adv_ = [0, 0, 0]
    #     elif args.test_sub == 1:
    #         ran = random.randint(0, 2)
    #         if ran == 0:
    #             obs_adv_ = [0, 0, 1]
    #         elif ran == 1:
    #             obs_adv_ = [0, 1, 0]
    #         else:
    #             obs_adv_ = [1, 0, 0]
    #     elif args.test_sub == 2:
    #         ran = random.randint(0, 2)
    #         if ran == 0:
    #             obs_adv_ = [0, 1, 1]
    #         elif ran == 1:
    #             obs_adv_ = [1, 1, 0]
    #         else:
    #             obs_adv_ = [1, 0, 1]
    #     else:  # args.test_sub == 3:
    #         obs_adv_ = [1, 1, 1]
    #
    #     obs_adv.append(obs_adv_)

    if args.mask_type == 'random':
        obs_adv_ = np.random.choice([0, 1], size=args.num_victim, p=[args.mask_rate, 1 - args.mask_rate])
        for _ in range(args.num_attacker):
            obs_adv.append(obs_adv_)

    elif args.mask_type == 'distance':
        distance_range = args.distance_range
        for i in range(args.num_agents):
            if i >= args.num_victim:
                obs_adv_ = np.ones(args.num_victim)
                for j in range(args.num_victim):
                    pos = 8 + j * 2
                    if sum(abs(obs_n[env_agents[i]][pos:pos + 2])) >= distance_range:
                        obs_adv_[j] = 0  # out of range agents are invisible
                obs_adv.append(obs_adv_)

    else:  # args.mask_type == 'region'
        for i in range(args.num_agents):
            if i >= args.num_victim:
                obs_adv_ = np.ones(args.num_victim)
                for j in range(args.num_victim):
                    pos = 8 + 2 * args.num_attacker + 2 * args.num_forests + j * 2
                    if abs(obs_n[env_agents[i]][pos]) > 0:
                        obs_adv_[j] = 1
                    else:
                        obs_adv_[j] = 0
                obs_adv.append(obs_adv_)

    for i in range(args.num_attacker):
        mask_adv = []

        if args.scenario == "predator_prey":
            env_judge = 0
        else:  # args.scenario == "world_comm"
            env_judge = 1

        # the design of the mask is determined by the environment
        mask_basic = np.ones(2 + 2 + 2 * 2 + (2 + 2) * env_judge)
        for j in range(args.num_victim):
            if obs_adv[i][j] == 1:
                mask_adv = np.concatenate((mask_adv, [1, 1]), axis=0)
            else:
                mask_adv = np.concatenate((mask_adv, [0, 0]), axis=0)

        mask_teammates = np.ones(1 * env_judge + 2 * (args.num_attacker - 1) + 2 * (args.num_attacker - 1))
        mask_ = np.concatenate((mask_basic, mask_adv, mask_teammates), axis=0)
        mask_ = mask_.astype(np.float32)
        mask.append(mask_)

    return mask, obs_adv

def partial_observation(args, env_agents, obs_n, mask):
    # region-based is automatically implemented by "world_comm"
    if args.mask_type == 'random' or args.mask_type == 'distance':
        k = 0
        for index in env_agents:
            if k >= args.num_victim:
                obs_n[index] = obs_n[index] * mask[k - args.num_victim]
            k += 1

    return obs_n

def stat_sub(args, sum_sub, obs_adv):
    for i in range(args.num_attacker):
        sub = sum(obs_adv[i])
        for j in range(args.num_victim + 1):
            if sub == j:
                sum_sub[j] += 1

    return sum_sub

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

def learn_judge(args, buffer_vic, buffer_att, num_learn, agents):
    if (buffer_vic.pointer - 1) >= args.memory_capacity and (buffer_vic.pointer - 1) % args.episode_limit == 0:

        if args.task == "train_all":
            agent_id = 0
            for agent in agents:
                if agent_id < args.num_victim:
                    agent.train(agents, buffer_vic, agent_id, 0)
                    num_learn += 1
                elif agent_id >= args.num_victim:
                    agent.train(agents, buffer_att, agent_id, 1)
                    num_learn += 1
                agent_id += 1

        elif args.task == "train_victim":
            agent_id = 0
            for agent in agents:
                if agent_id < args.num_victim:
                    agent.train(agents, buffer_vic, agent_id, 0)
                    num_learn += 1
                agent_id += 1

        elif args.task == "train_attacker":
            agent_id = 0
            for agent in agents:
                if agent_id >= args.num_victim:
                    agent.train(agents, buffer_att, agent_id, 1)
                    num_learn += 1
                agent_id += 1

    return num_learn

def policy_ensemble(args, agents, obs_n, obs_adv, store_rate):
    act = []
    for i in range(args.num_agents):
        if i < args.num_victim:
            sub_agents = agents[-1]
        else:
            if store_rate[int(sum(obs_adv[i - args.num_victim]))] == 0 and store_rate[0] == 0:
                sub_agents = agents[-1]
            elif store_rate[int(sum(obs_adv[i - args.num_victim]))] == 0 and store_rate[-1] == 0:
                sub_agents = agents[0]
            else:
                sub_agents = agents[int(sum(obs_adv[i - args.num_victim]))]

        act_ = sub_agents[i].choose_action(list(obs_n.values())[i], args.act_bound[i])
        act.append(act_)

    return act
