
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
    return info["balance"]

if __name__ == "__main__":
    env_name = "sku50.MultiStore.Standard"

    exp_name = "Ss_policy_S1.0_s1.0"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    balance = Ss_policy(env, [[1.0] * 50] * 3, [[1.0] * 50] * 3)
    env.render()
    print(balance)  # [42427.1625, 47639.125, 28938.5375]

    exp_name = "Ss_policy_S3.5_s3.5"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    balance = Ss_policy(env, [[3.5] * 50] * 3, [[3.5] * 50] * 3)
    env.render()
    print(balance)  # [539054.1, 615175.0125, 536784.15]

    exp_name = "Ss_policy_S4.0_s4.0"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    balance = Ss_policy(env, [[4.0] * 50] * 3, [[4.0] * 50] * 3)
    env.render()
    print(balance)  # [597919.8375, 684820.125, 592711.1375]

    exp_name = "Ss_policy_Ss_best"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test")
    balance = Ss_policy(env, [[3.5] * 50, [3.5] * 50, [4.0] * 50], [[3.0] * 50, [3.5] * 50, [4.0] * 50], vis_path=vis_path)
    env.render()
    print(balance)  # [589575.425, 666595.7, 585118.1125]