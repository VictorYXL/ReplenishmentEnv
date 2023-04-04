
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
    #mean_demand = env.agent_states["demand"]
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    re_fre = np.zeros(((env.warehouse_count, sku_count)))
    reward_info = {
        "profit": np.zeros(((env.warehouse_count, sku_count))),
        "excess_cost": np.zeros(((env.warehouse_count, sku_count))),
        "order_cost": np.zeros(((env.warehouse_count, sku_count))),
        "holding_cost": np.zeros(((env.warehouse_count, sku_count))),
        "backlog_cost": np.zeros(((env.warehouse_count, sku_count)))
    }
    while not done:
        mean_demand = env.get_demand_mean()
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        # 这个是统计每种SKU的补货数目
        re_fre += np.where(action < s, 1, 0)
        action = np.where(action < s, S - action, 0)
        state, reward, done, info = env.step(action)
        total_reward += reward
        reward_info["profit"] += info["reward_info"]["profit"]
        reward_info["excess_cost"] += info["reward_info"]["excess_cost"]
        reward_info["order_cost"] += info["reward_info"]["order_cost"]
        reward_info["holding_cost"] += info["reward_info"]["holding_cost"]
        reward_info["backlog_cost"] += info["reward_info"]["backlog_cost"]
    # TODO: 如果一个也没卖出，GMV=0怎么办？
    # 将GMV=0的在统计中去除
    GMV = np.sum(env.get_sale() * env.get_selling_price(), axis = 1)
    return total_reward, re_fre, GMV, reward_info
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
        print(S)
        for s in np.arange(0, S + 0.1, 0.5):
            rewards,_,_,_= Ss_policy(env, [[S] * sku_count]*env.warehouse_count, [[s] * sku_count]*env.warehouse_count)
            total_reward = np.sum(rewards)
            if total_reward > max_rewards:
                max_rewards = total_reward
                best_S = np.ones_like(best_S)*S
                best_s = np.ones_like(best_S)*s
    return max_rewards, best_S, best_s
#  现在要将其变成一个分层的best_S,best_s，即每层都有一个最好的S和s
#  经过测试，这样并搜索不出来好结果
def search_multilevel_all_independent_Ss(env, S_range=np.arange(4, 14.1, 2)):
    state = env.reset()
    #best_S = 0
    #best_s = 0
    sku_count   = len(env.get_sku_list())
    max_reward  = np.ones((env.warehouse_count, sku_count)) * (-np.inf)
    best_S      = np.zeros((env.warehouse_count, sku_count))
    best_s      = np.zeros((env.warehouse_count, sku_count))
    sku_count = len(env.get_sku_list())
    # S从6到18，s从4到12
    for S1 in S_range:
        for s1 in np.arange(4, min(S1 + 0.1, 12.1), 2):
            for S2 in S_range:
                for s2 in np.arange(4, min(S2 + 0.1, 12.1), 2):
                    for S3 in S_range:
                        for s3 in np.arange(4, min(S3 + 0.1, 12.1), 2):
                            S = [[S1] * sku_count] + [[S2] * sku_count] + [[S3] * sku_count]
                            s = [[s1] * sku_count] + [[s2] * sku_count] + [[s3] * sku_count]
                            print("S: {}, s:{}".format([S1,S2,S3],[s1,s2,s3]))
                            rewards,_,_,_= Ss_policy(env, S, s)
                            best_S          = np.where(rewards > max_reward, S, best_S)
                            best_s          = np.where(rewards > max_reward, s, best_s)
                            max_reward      = np.where(rewards > max_reward, rewards, max_reward)
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
                            rewards,_,_,_= Ss_policy(env, S, s)
                            total_reward = np.sum(rewards, axis = 0)
                            best_S[:, total_reward > max_reward] = S[:, total_reward > max_reward]
                            best_s[:, total_reward > max_reward] = s[:, total_reward > max_reward]
                            max_reward[total_reward >  max_reward] = total_reward[total_reward >  max_reward]
    return best_S, best_s

def search_independent_Ss(env, search_range=np.arange(0.0, 12.1, 1)):
    env.reset()
    sku_count   = len(env.get_sku_list())
    max_reward  = np.ones((sku_count)) * (-np.inf)
    best_S      = np.zeros((sku_count))
    best_s      = np.zeros((sku_count))
    
    for S in search_range:
        for s in np.arange(0, S + 0.1, 0.5):
            print([S,s])
            reward, _, _, _ = Ss_policy(env, [[S] * sku_count]*env.warehouse_count, [[s] * sku_count]*env.warehouse_count)
            best_S          = np.where(reward > max_reward, S, best_S)
            best_s          = np.where(reward > max_reward, s, best_s)
            max_reward      = np.where(reward > max_reward, reward, max_reward)
    return best_S, best_s

def analyze_Ss(env, best_S, best_s, output_file):
    env.reset()
    # GMV中有0元素，导致X的计算出现问题
    reward, re_fre, GMV, reward_info = Ss_policy(env, best_S, best_s)
    in_stock = env.agent_states["all_warehouses", "in_stock", "all_dates", "all_skus"].copy()
    max_in_stock_day = np.argmax(in_stock.sum(axis = (0, -1)))
    f = open(output_file+"__", "w")
    f.write("Warehouse,SKU,S,s,reward,profit,excess_cost,order_cost,holding_cost,backlog_cost,replenishment_times,GMV,X,max_in_stock_day\n")
    for warehouse in range(env.warehouse_count):
        for i in range(len(env.get_sku_list())):
            # TODO:将GMV为0的sku不写入到文件，让它不会干扰average X的计算
            #if GMV[warehouse, i] > 1e-5:
            if GMV[warehouse, i] < 1e-5:
                continue
            f.write(
                str(warehouse + 1) + "," \
                + env.get_sku_list()[i] + "," \
                + str(best_S[warehouse, i]) + "," \
                + str(best_s[warehouse, i]) + "," \
                + str(reward[warehouse, i]) + "," \
                + str(reward_info["profit"][warehouse, i]) + "," \
                + str(reward_info["excess_cost"][warehouse, i]) + "," \
                + str(reward_info["order_cost"][warehouse, i]) + "," \
                + str(reward_info["holding_cost"][warehouse, i]) + "," \
                + str(reward_info["backlog_cost"][warehouse, i]) + "," \
                + str(re_fre[warehouse, i]) + ","\
                + str(GMV[warehouse, i]) + ","\
                + str(reward_info["holding_cost"][warehouse, i]* 365 / GMV[warehouse, i]) + ","\
                + str(in_stock[warehouse,max_in_stock_day,i]) + "\n"
            )
    f.close()

def get_task_list():
    #sku_list = ["50", "100", "200", "500", "1000", "2307"]
    # challenge_list = [
    #     "Standard",
    #     "BacklogRatioHigh", "BacklogRatioLow", "BacklogRatioMiddle", 
    #     "CapacityHigh", "CapacityLow", "CapacityLowest",
    #     "OrderCostHigh", "OrderCostHighest", "OrderCostLow",
    #     "StorageCostHigh", "StorageCostHighest", "StorageCostLow"
    # ]
    # sku_list = ["50"]
    sku_list = ["1000"]
    store_list = ["multi_store"]
    challenge_list = ["standard copy 8","standard copy 7","standard copy 6","standard copy 5","standard copy 4","standard copy 3","standard copy 2","standard copy 1","standard"]
    task_list = []
    for sku in sku_list:
        for store in store_list:
            for challenge in challenge_list:
                task_list.append("sku" + sku + "." + store + "." + challenge)
    return task_list

def summary(input_dir, output_file):
    f = open(output_file, "w")
    
    f.write("Task,Mode,Total_Reward,Total_Profit,Total_Excess,Total_Order_Cost,Total_Holding_Cost,Total_Backlog,X<0.1,X>0.25,Average_X,GMV,Max_in_stock,Replenishment_frq,s=0,S>=12,Average_S,Average_s\n")
    for name in os.listdir(input_dir):
        file_name = os.path.join(input_dir, name)
        sku_count = int(name.split('.')[0][3:])
        data = []
        df = pd.read_csv(file_name, sep=",").fillna(0)
        data.append('.'.join(name.split('.')[:-2]))
        data.append(name.split('.')[-2])
        data.append(str(np.round(np.sum(df["reward"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["profit"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["excess_cost"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["order_cost"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["holding_cost"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["backlog_cost"]) / 1e3, 2)) + "K")
        data.append(str(len(df[df["X"].astype(float) < 0.1])))
        data.append(str(len(df[df["X"].astype(float) > 0.25])))
        data.append(str(np.round(np.average(df["X"]), 2)))
        data.append(str(np.round(np.sum(df["GMV"]) / 1e3, 2)))
        data.append(str(np.sum(df["max_in_stock_day"])))
        data.append(str(np.round(sku_count * 100 / np.sum(df["replenishment_times"]), 2)))
        data.append(str(len(df[df["s"] == 0])))
        data.append(str(len(df[df["s"] >= 12])))
        data.append(str(np.round(np.average(df["S"]), 2)))
        data.append(str(np.round(np.average(df["s"]), 2)))
        f.write(",".join(data) + "\n")
    f.close()

if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.join("output_multilevel", "different_Ss_shared_100000"), exist_ok=True)
    output_dir = os.path.join("output_multilevel", "different_Ss_shared_100000")

    task_list = get_task_list()
    # mode_list = ["validation", "test"]
    mode_list = ["test"]
    
    for task in task_list:
        for mode in mode_list:
            print(task)
            env = make_env(task, wrapper_names=["OracleWrapper"], mode='test')
            #env.reset()
            # best_S, best_s = search_independent_Ss(env)
            max_rewards, best_S, best_s = search_shared_Ss(env)
            # best_S, best_s = search_multilevel_interlevel_independent_Ss(env)

            analyze_Ss(env, best_S, best_s, os.path.join(output_dir, task + "." + mode + ".csv"))
    
    #summary(output_dir, os.path.join("output_multilevel", "different_Ss_shared_100000.summary.csv"))