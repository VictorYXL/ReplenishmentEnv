
import os
import numpy as np
import sys
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../..")
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
        print(S)
        for s in np.arange(0, S + 0.1, 1):
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
    

def get_ranks(my_array):
    # my_array = np.array([7, 2, 5, 1, 9])

    # 使用argsort()函数获取元素排序后的索引数组
    sorted_indices = np.argsort(my_array)

    # 创建一个与原始数组相同大小的数组，用于存储每个元素的排名
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(my_array)) + 1
    return ranks

def sS_search_different_profit(env_name, vis_path):
    """(s, S) algorithm hindsight mode."""
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test')
    best_S, best_s = search_sS(env_train)
    demand = env_train.agent_states["all_warehouses", "demand","all_dates"]
    selling_price = env_train.agent_states["all_warehouses", "selling_price","all_dates"]
    procurement_cost = env_train.agent_states["all_warehouses", "procurement_cost","all_dates"]
    maximum_possible_profit = (selling_price - procurement_cost) * demand
    # 得到了最大可能利润的排序，然后根据排序比较接近的，来计算最大距离
    sorted_maximum_possible_profit = np.argsort(-1 * maximum_possible_profit)
    max_rank_distance = -1
    # for i in range()


    min_s = np.min(best_s)
    max_s = np.max(best_s)
    min_S = np.min(best_S)
    max_S = np.max(best_S)
    
    # sorted_s = np.argsort(best_s)
    # # S1, S2都是位置，不是具体的值
    # select_SKU1, select_SKU2 = None, None
    # for i in range(len(sorted_s)):
    #     if best_s[sorted_s[i]] != min_s:
    #         break
    #     if select_SKU1 == None:
    #         select_SKU1 = sorted_s[i]
    #         continue
    #     if select_SKU2 == None:
    #         select_SKU2 = sorted_s[i]
    #         continue
    #     delta = abs(best_S[select_SKU2] - best_S[select_SKU1])
    #     if abs(best_S[sorted_s[i]] - best_S[select_SKU1]) > delta:
    #         select_SKU2 = sorted_s[i]
    #     elif abs(best_S[sorted_s[i]] - best_S[select_SKU2]) > delta:
    #         select_SKU1 = sorted_s[i]
    # print(select_SKU1)
    # print(select_SKU2)
    
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test', vis_path=vis_path)
    _, balance = sS_policy(env_test, best_S, best_s)
    # a = _.sum(axis = 0)
    # print("least 10 SKU:")
    # print(np.argsort(a)[:10])
    # print("highest 10 SKU:")
    # print(np.argsort(-a)[:10])
    # env_test.render()
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

if __name__ == "__main__":
    env_name = "sku200.2_stores.standard"
    exp_name = "search_sS"
    vis_path = os.path.join("output", exp_name)
    # sS_search_different_profit(env_name, vis_path)
    env = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
    env.reset()
    balance = sS_hindsight(env_name=env_name,vis_path=vis_path)
    print(np.sum(balance))