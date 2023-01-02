
import os
import numpy as np
import sys
import pandas as pd

env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)

from ReplenishmentEnv import make_env

def Ss_policy(env, S, s, exp_name=None):
    env.reset(exp_name)
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

    exp_name = "Ss_policy_S1.0_s1.0"
    rewards = Ss_policy(env, [[1.0] * 50] * 3, [[1.0] * 50] * 3, exp_name)
    print(np.sum(rewards, 1))
    env.render()

    exp_name = "Ss_policy_S3.5_s3.5"
    rewards = Ss_policy(env, [[3.5] * 50] * 3, [[3.5] * 50] * 3, exp_name)
    print(np.sum(rewards, 1))
    env.render()

    exp_name = "Ss_policy_S4.0_s4.0"
    rewards = Ss_policy(env, [[4.0] * 50] * 3, [[4.0] * 50] * 3, exp_name)
    print(np.sum(rewards, 1))
    env.render()

    exp_name = "Ss_policy_Ss_best"
    rewards = Ss_policy(env, [[3.5] * 50, [3.5] * 50, [4.0] * 50], [[3.0] * 50, [3.5] * 50, [4.0] * 50], exp_name)
    print(np.sum(rewards, 1))
    env.render()