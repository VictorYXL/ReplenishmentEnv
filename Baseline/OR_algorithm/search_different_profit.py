import random
import sys
import os
import numpy as np
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../..")
sys.path.insert(0, env_dir)
from ReplenishmentEnv import make_env
def get_ranks(my_array):
    # my_array = np.array([7, 2, 5, 1, 9])

    # 使用argsort()函数获取元素排序后的索引数组
    sorted_indices = np.argsort(my_array)

    # 创建一个与原始数组相同大小的数组，用于存储每个元素的排名
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(my_array)) + 1
    return ranks

def search_different_profit(env_train):
    """(s, S) algorithm hindsight mode."""
    # env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test')
    # best_S, best_s = search_sS(env_train)
    demand = env_train.agent_states["all_warehouses", "demand","all_dates"][:, 21:]
    sku_total_demand = demand.sum(axis = 1).reshape(-1)
    selling_price = env_train.agent_states["all_warehouses", "selling_price","all_dates"][:, 21:]
    rank_demand = get_ranks(demand.sum(axis = 1).reshape(-1))
    # rank_demand = get_ranks(demand)
    procurement_cost = env_train.agent_states["all_warehouses", "procurement_cost","all_dates"][:, 21:]
    maximum_possible_profit = ((selling_price - procurement_cost) * demand).sum(axis = 1).reshape(-1)
    # 得到了最大可能利润的排序，然后根据排序比较接近的，来计算最大距离
    sorted_maximum_possible_profit = np.argsort(-1 * maximum_possible_profit)
    max_rank_distance = -1
    demand_limit = 500
    sku1, sku2 = 0, 1
    # search for SKU with similar profit and different demand
    # for i in range(195):
    #     for j in range(i+1,i+5):
    #         rank_distance = abs(rank_demand[sorted_maximum_possible_profit[i]] - rank_demand[sorted_maximum_possible_profit[j]])
    #         if rank_distance > max_rank_distance:
    #             max_rank_distance = rank_distance
    #             sku1, sku2 = sorted_maximum_possible_profit[i],sorted_maximum_possible_profit[j]
    #             print(i,j)
    #             print("new rank distance : {}, sku1 : {}, profit : {}, demand : {}, sku2 : {}, profit : {}, demand : {}".format(max_rank_distance,sku1,maximum_possible_profit[sku1],sku_total_demand[sku1],sku2,maximum_possible_profit[sku2],sku_total_demand[sku2]))
    # search for SKU with high profit and low demand
    for i in range(195):
        for j in range(i+1,i+5):
            demand1, demand2 = sku_total_demand[sorted_maximum_possible_profit[i]], sku_total_demand[sorted_maximum_possible_profit[j]]
            if demand1 < 500 and demand2 < 500:
                sku1, sku2 = sorted_maximum_possible_profit[i],sorted_maximum_possible_profit[j]
                print("sku1 : {}, profit : {}, demand : {}, sku2 : {}, profit : {}, demand : {}".format(sku1,maximum_possible_profit[sku1],sku_total_demand[sku1],sku2,maximum_possible_profit[sku2],sku_total_demand[sku2]))
                
    
    
if __name__ == "__main__":
    env_name = "sku200.single_store.standard"
    exp_name = "random_action"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["OracleWrapper"], mode='test')
    env.reset()
    search_different_profit(env)
    # for i in range(10):
    #     action_list = [[random.random() * 10 for i in range(1000)] for j in range(3)]
    #     states, rewards, done, info_states = env.step(action_list) 
    # print(info_states["balance"])