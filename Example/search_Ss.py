
import os
import numpy as np
import sys
import pandas as pd

env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)

from ReplenishmentEnv import make_env

def Ss_policy(env, S, s):
    env.reset()
    done = False
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    while not done:
        mean_demand = env.get_demand_mean()
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        action = np.where(action < s, S - action, 0)
        state, reward, done, info = env.step(action)
        total_reward += reward
    return info["balance"]

def search_shared_Ss(env, S_range=np.arange(0.0, 12.1, 1)):
    state = env.reset()
    max_rewards = -np.inf
    #best_S = 0
    #best_s = 0
    sku_count   = len(env.get_sku_list())
    best_S      = np.zeros((env.warehouse_count, sku_count))
    best_s      = np.zeros((env.warehouse_count, sku_count))
    sku_count = len(env.get_sku_list())
    for S in S_range:
        for s in np.arange(0, S + 0.1, 0.5):
            rewards,_,_,_= Ss_policy(env, [[S] * sku_count]*env.warehouse_count, [[s] * sku_count]*env.warehouse_count)
            total_reward = np.sum(rewards)
            if total_reward > max_rewards:
                max_rewards = total_reward
                best_S = np.ones_like(best_S)*S
                best_s = np.ones_like(best_S)*s
    return max_rewards, best_S, best_s

def search_independent_Ss(env, search_range=np.arange(0.0, 12.1, 1)):
    env.reset()
    sku_count   = len(env.get_sku_list())
    max_reward  = np.ones((sku_count)) * (-np.inf)
    best_S      = np.zeros((sku_count))
    best_s      = np.zeros((sku_count))
    
    for S in search_range:
        for s in np.arange(0, S + 0.1, 0.5):
            reward          = Ss_policy(env, [[S] * sku_count]*env.warehouse_count, [[s] * sku_count]*env.warehouse_count)
            best_S          = np.where(reward > max_reward, S, best_S)
            best_s          = np.where(reward > max_reward, s, best_s)
            max_reward      = np.where(reward > max_reward, reward, max_reward)
    return best_S, best_s

def search_multilevel_interlevel_independent_Ss(env, S_range=np.arange(6, 12.1, 2)):
    state = env.reset()
    #best_S = 0
    #best_s = 0
    sku_count   = len(env.get_sku_list())
    # max_reward  = np.ones((env.warehouse_count, sku_count)) * (-np.inf)
    max_reward  = np.ones(sku_count) * (-np.inf)
    best_S      = np.zeros((env.warehouse_count, sku_count))
    best_s      = np.zeros((env.warehouse_count, sku_count))
    sku_count = len(env.get_sku_list())
    # S从6到18，s从4到12
    for S1 in S_range:
        for s1 in np.arange(2, min(S1 + 0.1, 12.1), 2):
            for S2 in S_range:
                for s2 in np.arange(2, min(S2 + 0.1, 12.1), 2):
                    for S3 in S_range:
                        for s3 in np.arange(2, min(S3 + 0.1, 12.1), 2):
                            S = np.array([[S1] * sku_count] + [[S2] * sku_count] + [[S3] * sku_count])
                            s = np.array([[s1] * sku_count] + [[s2] * sku_count] + [[s3] * sku_count])
                            print("S: {}, s:{}".format([S1,S2,S3],[s1,s2,s3]))
                            rewards = Ss_policy(env, S, s)
                            total_reward = np.sum(rewards, axis = 0)
                            best_S[:, total_reward > max_reward] = S[:, total_reward > max_reward]
                            best_s[:, total_reward > max_reward] = s[:, total_reward > max_reward]
                            max_reward[total_reward >  max_reward] = total_reward[total_reward >  max_reward]
    return best_S, best_s

def Ss_oracle_independent(env_name):
    exp_name = "Ss_oracle_independent"
    vis_path = os.path.join("output", env_name, exp_name)
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test')
    # best_S, best_s = search_multilvel_interlevel_independent_Ss(env)
    if env_train.warehouse_count == 1:
        best_S, best_s = search_independent_Ss(env_train)
    else:
        best_S, best_s = search_multilevel_interlevel_independent_Ss(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test', vis_path=vis_path)
    balance = Ss_policy(env_test, best_S, best_s)
    env_test.render()
    print(vis_path, balance)

def Ss_static_independent(env_name):
    exp_name = "Ss_static_independent"
    vis_path = os.path.join("output", env_name, exp_name)
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='train')
    if env_train.warehouse_count == 1:
        best_S, best_s = search_independent_Ss(env_train)
    else:
        best_S, best_s = search_multilevel_interlevel_independent_Ss(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test', vis_path=vis_path)
    balance = Ss_policy(env_test, best_S, best_s)
    env_test.render()
    print(vis_path, balance)