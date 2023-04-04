
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

def read_S_s_from_file(env, path):
    sku_list = env.get_sku_list()
    df = pd.read_csv(path)
    S = [[] for i in range(env.warehouse_count)]
    s = [[] for i in range(env.warehouse_count)]
    for warehouse in range(1, env.warehouse_count+1):
        df_warehouse = df[df['Warehouse'] == warehouse]
        df_warehouse.index = df_warehouse['SKU']
        for sku in sku_list:
            if sku in df_warehouse['SKU'].index:
                S[warehouse-1].append(df_warehouse['S'].loc[sku])
                s[warehouse-1].append(df_warehouse['s'].loc[sku])
            else:
                # 如果文件里没这个元素，那么就说明他没什么需求，因此才没卖出的。因此我们设置S=0s=0,不进货就可以了
                S[warehouse-1].append(0)
                s[warehouse-1].append(0)
    return S,s


if __name__ == "__main__":
    env_name = "sku1000.multi_store.standard copy 4"
    exp_name = "standard copy 4 2000000"
    vis_path = os.path.join("output", env_name, exp_name)
    env = make_env(env_name, wrapper_names=["DynamicWrapper"], mode="test", vis_path=vis_path)
    S, s = read_S_s_from_file(env, "./output_multilevel/different_Ss/sku1000.multi_store.standard copy 4.test.csv__")
    balance = Ss_policy(env, S, s)
    env.render()
    print(env_name, exp_name, balance)
    # env_names = [
    #     "sku50.multi_store.standard",
    #     "sku100.multi_store.standard",
    #     "sku200.multi_store.standard"
    #     "sku500.multi_store.standard",
    #     "sku1000.multi_store.standard",
    #     "sku2000.multi_store.standard",
    # ]
    
    # for env_name in env_names:
    #     exp_name = "Ss_policy_S1.0_s1.0"
    #     vis_path = os.path.join("output", env_name, exp_name)
    #     env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    #     balance = Ss_policy(env, [[1.0] * env.sku_count] * env.warehouse_count, [[1.0] * env.sku_count] * env.warehouse_count)
    #     env.render()
    #     print(env_name, exp_name, balance)

    #     exp_name = "Ss_policy_S3.0_s3.0"
    #     vis_path = os.path.join("output", env_name, exp_name)
    #     env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    #     balance = Ss_policy(env, [[3.0] * env.sku_count] * env.warehouse_count, [[3.0] * env.sku_count] * env.warehouse_count)
    #     env.render()
    #     print(env_name, exp_name, balance)

    #     exp_name = "Ss_policy_Ss_S5.0_s5.0"
    #     vis_path = os.path.join("output", env_name, exp_name)
    #     env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    #     balance = Ss_policy(env, [[5.0] * env.sku_count] * env.warehouse_count, [[5.0] * env.sku_count] * env.warehouse_count)
    #     env.render()
    #     print(env_name, exp_name, balance)

    #     exp_name = "Ss_policy_Ss_S7.0_s7.0"
    #     vis_path = os.path.join("output", env_name, exp_name)
    #     env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    #     balance = Ss_policy(env, [[7.0] * env.sku_count] * env.warehouse_count, [[7.0] * env.sku_count] * env.warehouse_count)
    #     env.render()
    #     print(env_name, exp_name, balance)