
import os
import numpy as np
import sys
import pandas as pd

env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)

from ReplenishmentEnv import make_env

def sS_policy(env, S, s):
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
    return total_reward, info["balance"]

def search_sS(env, search_range=np.arange(0.0, 12.1, 1)):
    env.reset()
    sku_count   = len(env.get_sku_list())
    max_reward  = np.ones((sku_count)) * (-np.inf)
    best_S      = np.zeros((sku_count))
    best_s      = np.zeros((sku_count))
    
    for S in search_range:
        for s in np.arange(0, S + 0.1, 0.5):
            reward, _       = sS_policy(env, [[S] * sku_count]*env.warehouse_count, [[s] * sku_count]*env.warehouse_count)
            reward      = sum(reward)
            best_S          = np.where(reward > max_reward, S, best_S)
            best_s          = np.where(reward > max_reward, s, best_s)
            max_reward      = np.where(reward > max_reward, reward, max_reward)
    return np.ones((env.warehouse_count, sku_count)) * best_S, np.ones((env.warehouse_count, sku_count)) * best_s

def sS_hindsight(env_name, vis_path):
    """(s, S) algorithm hindsight mode."""
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test')
    best_S, best_s = search_sS(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test', vis_path=vis_path)
    _, balance = sS_policy(env_test, best_S, best_s)
    env_test.render()
    return balance

def sS_static(env_name, vis_path):
    """(s, S) algorithm static mode.
    The random_interception in config file will add an element of randomness.
    """
    exp_name = "sS_static"
    vis_path = os.path.join("output", env_name, exp_name)
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='train')
    best_S, best_s = search_sS(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test', vis_path=vis_path)
    _, balance = sS_policy(env_test, best_S, best_s)
    env_test.render()
    return balance