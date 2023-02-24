
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
    warehouse_count = len(env.warehouse_list)
    rewards = np.zeros((warehouse_count, sku_count))
    while not done:
        mean_demand = env.get_demand_mean()
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        action = np.where(action < s, S - action, 0)
        state, reward, done, info = env.step(action)
        rewards += reward
    return info["balance"]

if __name__ == "__main__":
    env_names = [
        "sku50.multi_store.standard",
        "sku2000.multi_store.standard",
        "sku1000.multi_store.standard",
        "sku500.multi_store.standard",
        "sku100.multi_store.standard",
        "sku200.multi_store.standard"
    ]
    
    for env_name in env_names:
        exp_name = "Ss_policy_S1.0_s1.0"
        vis_path = os.path.join("output", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
        balance = Ss_policy(env, [[2.0] * env.sku_count] * env.warehouse_count, [[2.0] * env.sku_count] * env.warehouse_count)
        env.render()
        print(env_name, exp_name, balance)

        exp_name = "Ss_policy_S3.0_s3.0"
        vis_path = os.path.join("output", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
        balance = Ss_policy(env, [[3.0] * env.sku_count] * env.warehouse_count, [[3.0] * env.sku_count] * env.warehouse_count)
        env.render()
        print(env_name, exp_name, balance)

        exp_name = "Ss_policy_Ss_S5.0_s5.0"
        vis_path = os.path.join("output", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
        balance = Ss_policy(env, [[5.0] * env.sku_count] * env.warehouse_count, [[5.0] * env.sku_count] * env.warehouse_count)
        env.render()
        print(env_name, exp_name, balance)

        exp_name = "Ss_policy_Ss_S7.0_s7.0"
        vis_path = os.path.join("output", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
        balance = Ss_policy(env, [[7.0] * env.sku_count] * env.warehouse_count, [[7.0] * env.sku_count] * env.warehouse_count)
        env.render()
        print(env_name, exp_name, balance)