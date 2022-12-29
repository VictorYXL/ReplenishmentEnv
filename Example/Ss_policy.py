
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
    facility_count = len(env.facility_list)
    rewards = np.zeros((facility_count, sku_count))
    while not done:
        mean_demand = env.get_demand_mean()
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        action = np.where(action < s, S - action, 0)
        state, reward, done, info = env.step(action)
        rewards += reward
    return rewards

if __name__ == "__main__":
    env_name = "sku50.MultiStore.Standard"
    env = make_env(env_name, "DefaultWrapper", "test")
    rewards = Ss_policy(env, [[4.0] * 50] * 3, [[4.0] * 50] * 3)
    print(np.sum(rewards, 1))   # 461294.8508 679401.6854 606858.4326]
    rewards = Ss_policy(env, [[3.5] * 50] * 3, [[3.5] * 50] * 3)
    print(np.sum(rewards, 1))   # 518701.9036 617936.1934 556545.9074
    rewards = Ss_policy(env, [[3.0] * 50] * 3, [[3.0] * 50] * 3)
    print(np.sum(rewards, 1))   # 449359.4684 526129.107  485747.0012
    rewards = Ss_policy(env, [[3.5] * 50, [3.5] * 50, [4.0] * 50], [[3.0] * 50, [3.5] * 50, [4.0] * 50])
    print(np.sum(rewards, 1))   # 570439.9032 680407.048  599142.1822