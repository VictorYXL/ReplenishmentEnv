import argparse
import os
import sys
import gym
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)

from ReplenishmentEnv import make_env

def get_multilevel_stock_level(env: gym.Wrapper, env_mode = "test", profit_include_t0 = False):
    stock_levels = np.zeros((len(env.get_warehouse_list()), len(env.get_sku_list())))
    for sku_index, sku in enumerate(env.get_sku_list()):
        selling_price = np.array(env.get_selling_price(sku = sku))
        procurement_cost = np.array(env.get_procurement_cost(sku = sku))
        demand = np.array(env.get_demand(sku = sku))
        average_vlt = env.get_average_vlt(sku = sku).max()
        in_stock = np.array(env.get_in_stock(sku = sku))
        holding_cost = np.array(env.get_holding_cost(sku = sku))
        replenishment_before = np.array(env.get_replenishment_before(sku = sku))
        if demand.ndim == 2:
            demand = demand[-1].reshape(-1)
        if env_mode == "train":
            stock_level = get_multilevel_single_stock_level(
                selling_price,
                procurement_cost,
                demand,
                average_vlt,
                in_stock,
                holding_cost,
                len(env.get_warehouse_list()),
                None,
                True
            ).reshape(-1,)
        else:
            stock_level = get_multilevel_single_stock_level(
                selling_price,
                procurement_cost,
                demand,
                average_vlt,
                in_stock,
                holding_cost,
                len(env.get_warehouse_list()),
                replenishment_before,
                False
            ).reshape(-1,)
        stock_levels[:, sku_index] = stock_level
    return stock_levels

def get_multilevel_single_stock_level(
        selling_price: np.array,
        procurement_cost: np.array,
        demand: np.array,
        vlt: int,
        in_stock: np.array,
        holding_cost: np.array,
        warehouses: int,
        replenishment_before: np.array = None,
        profit_include_t0 = False,
) -> np.ndarray:
    if isinstance(replenishment_before, np.ndarray):
        len_replenishment_before = replenishment_before.shape[-1]
    time_hrz_len = len(selling_price[0])
    stocks = cp.Variable([warehouses, time_hrz_len + 1], integer=True)
    transits = cp.Variable([warehouses, time_hrz_len + 1], integer=True)
    sales = cp.Variable([warehouses, time_hrz_len], integer=True)
    buy = cp.Variable([warehouses, time_hrz_len + vlt], integer=True)
    buy_in = cp.Variable([warehouses, time_hrz_len], integer=True)
    buy_arv = cp.Variable([warehouses, time_hrz_len], integer=True)
    stock_level = cp.Variable([warehouses, 1], integer=True)
    profit = cp.Variable(1)
    common_constraints = [
        stocks >= 0,
        transits >= 0,
        sales >= 0,
        buy >= 0
    ]
    intralevel_constraints = [
        stocks[:, 0] == in_stock,
        stocks[:, 1: time_hrz_len + 1] == stocks[:, 0:time_hrz_len] + buy_arv - sales,
        transits[:, 1: time_hrz_len + 1] == transits[:, 0:time_hrz_len] - buy_arv + buy_in,
        sales <= stocks[:, 0:time_hrz_len],
        buy_in == buy[:, vlt: time_hrz_len + vlt],
        buy_arv == buy[:, 0:time_hrz_len],
        stock_level == stocks[:, 0:time_hrz_len] + transits[:, 0:time_hrz_len] + buy_in,
    ]
    if profit_include_t0:
        intralevel_constraints.append(
            profit == cp.sum(
                cp.multiply(selling_price, sales) - cp.multiply(procurement_cost, buy_in) - cp.multiply(holding_cost,
                                                                                                        stocks[:, 1:]),
            ) - cp.sum(cp.multiply(procurement_cost[:, 0], transits[:, 0]))
        )
    else:
        intralevel_constraints.append(
            profit == cp.sum(
                cp.multiply(selling_price, sales) - cp.multiply(procurement_cost, buy_in) - cp.multiply(holding_cost,
                                                                                                        stocks[:, 1:]),
            )
        )
    if isinstance(replenishment_before, np.ndarray):
        intralevel_constraints.append(transits[:, 0] == np.sum(replenishment_before, axis = 1))
        if len_replenishment_before>0:
            intralevel_constraints.append(buy[:, :len_replenishment_before] == replenishment_before)
    interlevel_constraints = []
    for i in range(warehouses):
        if i != warehouses-1:
            interlevel_constraints.append(sales[i] == buy_in[i+1])
        else:
            interlevel_constraints.append(sales[i] <= demand)
    
    constraints = common_constraints + intralevel_constraints + interlevel_constraints
    obj = cp.Maximize(profit)
    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.SCIP, verbose=False)
    if prob.status != 'optimal':
        prob.solve(solver=cp.GLPK_MI, verbose=False, max_iters = 1000)
    # if prob.status != 'optimal':
    assert prob.status == 'optimal', 'can\'t find optimal solution for SKU stock level'


    # prob.solve(solver=cp.GLPK_MI, verbose=False)
    return stock_level.value

def multilevel_base_stock(env: gym.Wrapper, update_freq =7, static_stock_levels = None):
    env.reset()
    current_step = 0
    is_done = False
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    stock_level_list = [[] for i in range(len(env.get_warehouse_list()))]
    reward_info = {
        "profit": np.zeros(((env.warehouse_count, sku_count))),
        "excess_cost": np.zeros(((env.warehouse_count, sku_count))),
        "order_cost": np.zeros(((env.warehouse_count, sku_count))),
        "holding_cost": np.zeros(((env.warehouse_count, sku_count))),
        "backlog_cost": np.zeros(((env.warehouse_count, sku_count)))
    }
    while not is_done:
        if current_step % update_freq == 0:
            if isinstance(static_stock_levels, np.ndarray):
                stock_levels = static_stock_levels
            else:
                stock_levels = get_multilevel_stock_level(env)
        for i in range(len(env.get_warehouse_list())):
            stock_level_list[i].append(stock_levels[i])

        replenish = stock_levels - env.get_in_stock() - env.get_in_transit()
        replenish = np.where(replenish >= 0, replenish, 0) / (env.get_demand_mean() + 0.00001)
        states, reward, is_done, info = env.step(replenish)
        total_reward += reward
        reward_info["profit"] += info["reward_info"]["profit"]
        reward_info["excess_cost"] += info["reward_info"]["excess_cost"]
        reward_info["order_cost"] += info["reward_info"]["order_cost"]
        reward_info["holding_cost"] += info["reward_info"]["holding_cost"]
        reward_info["backlog_cost"] += info["reward_info"]["backlog_cost"]
        current_step += 1

    sale = env.agent_states["all_warehouses", "sale", slice(env.lookback_len, None, None), "all_skus"].copy()
    selling_price = env.agent_states["all_warehouses", "selling_price", slice(env.lookback_len, None, None), "all_skus"].copy()
    GMV = np.sum(sale * selling_price, axis = 1)
    return info["balance"], stock_level_list, GMV, total_reward, reward_info


def analyze(env, reward, GMV, reward_info, output_file):
    in_stock = env.agent_states["all_warehouses", "in_stock", "all_dates", "all_skus"].copy()
    max_in_stock_day = np.argmax(in_stock.sum(axis = (0, -1)))
    f = open(output_file, "w")
    f.write("Warehouse,SKU,reward,profit,excess_cost,order_cost,holding_cost,backlog_cost,GMV,X,max_in_stock_day\n")
    for warehouse in range(env.warehouse_count):
        for i in range(len(env.get_sku_list())):
            # If there is no sales of a certain type of SKU, we don't include it in the statistics
            if GMV[warehouse, i] < 1e-5:
                continue
            f.write(
                str(warehouse + 1) + "," \
                + env.get_sku_list()[i] + "," \
                + str(reward[warehouse, i]) + "," \
                + str(reward_info["profit"][warehouse, i]) + "," \
                + str(reward_info["excess_cost"][warehouse, i]) + "," \
                + str(reward_info["order_cost"][warehouse, i]) + "," \
                + str(reward_info["holding_cost"][warehouse, i]) + "," \
                + str(reward_info["backlog_cost"][warehouse, i]) + "," \
                + str(GMV[warehouse, i]) + ","\
                + str(reward_info["holding_cost"][warehouse, i]* 365 / GMV[warehouse, i]) + ","\
                + str(in_stock[warehouse,max_in_stock_day,i]) + "\n"
            )
    f.close()

def summary(file_name, output_file):
    f = open(output_file, "w")
    
    f.write("Total_Reward,Total_Profit,Total_Excess,Total_Order_Cost,Total_Holding_Cost,Total_Backlog,X<0.1,X>0.25,Average_X,GMV,Max_in_stock\n")
    data = []
    df = pd.read_csv(file_name)
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
    f.write(",".join(data) + "\n")
    f.close()

if __name__ == "__main__":
    env_names = [
        "sku50.multi_store.standard",
        "sku100.multi_store.standard",
        "sku200.multi_store.standard"
        "sku500.multi_store.standard",
        "sku1000.multi_store.standard",
        "sku2000.multi_store.standard",
    ]
    update_config = {"start_date": "2018/8/1", "end_date": "2018/9/31"}
    # update_config = None
    for env_name in env_names:
        exp_name = "oracle"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path, update_config=update_config)
        env_train.reset()
        static_stock_levels = get_multilevel_stock_level(env_train, env_mode='train')
        env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        balance, oracle_stock_levels_list, GMV, reward, reward_info = multilevel_base_stock(env_test, static_stock_levels = static_stock_levels)
        os.makedirs(vis_path, exist_ok=True)
        analyze(env_test, reward, GMV, reward_info, os.path.join(vis_path, 'analysis.csv'))
        summary(os.path.join(vis_path, 'analysis.csv'), os.path.join(vis_path,'summary.csv'))
        env_test.render()
        print(env_name, exp_name, balance)
        print(np.mean(static_stock_levels,axis=1))

        exp_name = "static"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="train", vis_path=vis_path, update_config=update_config)
        env_train.reset()
        # no warmup so the env_mode here is test
        static_stock_levels = get_multilevel_stock_level(env_train, env_mode = 'train')
        env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        balance, static_stock_levels_list, GMV, reward, reward_info  = multilevel_base_stock(env_test, static_stock_levels = static_stock_levels)
        os.makedirs(vis_path, exist_ok=True)
        analyze(env_test, reward, GMV, reward_info, os.path.join(vis_path, 'analysis.csv'))
        summary(os.path.join(vis_path, 'analysis.csv'), os.path.join(vis_path,'summary.csv'))
        env_test.render()
        print(env_name, exp_name, balance)
        print(np.mean(static_stock_levels,axis=1))

        exp_name = "dynamic_history"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["StaticWrapper"], mode="test", vis_path=vis_path)
        balance, history_stock_levels_list, GMV, reward, reward_info = multilevel_base_stock(env)
        os.makedirs(vis_path, exist_ok=True)
        analyze(env, reward, GMV, reward_info, os.path.join(vis_path, 'analysis.csv'))
        summary(os.path.join(vis_path, 'analysis.csv'), os.path.join(vis_path,'summary.csv'))
        env.render()
        print(env_name, exp_name, balance)

        exp_name = "dynamic_21"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["DynamicWrapper"], mode="test", vis_path=vis_path)
        balance, lookback21_stock_levels_list, GMV, reward, reward_info  = multilevel_base_stock(env)
        os.makedirs(vis_path, exist_ok=True)
        analyze(env, reward, GMV, reward_info, os.path.join(vis_path, 'analysis.csv'))
        summary(os.path.join(vis_path, 'analysis.csv'), os.path.join(vis_path,'summary.csv'))
        env.render()
        print(env_name, exp_name, balance)

        oracle_stock_levels_list = np.array(oracle_stock_levels_list)
        static_stock_levels_list = np.array(static_stock_levels_list)
        history_stock_levels_list = np.array(history_stock_levels_list)
        lookback21_stock_levels_list = np.array(lookback21_stock_levels_list)

        fig,ax = plt.subplots(3,1)
        for i in range(len(env.get_warehouse_list())):
            ax[i].plot(oracle_stock_levels_list[i].sum(axis = -1))
            ax[i].plot(static_stock_levels_list[i].sum(axis = -1))
            ax[i].plot(history_stock_levels_list[i].sum(axis = -1))
            ax[i].plot(lookback21_stock_levels_list[i].sum(axis = -1))
            ax[i].legend(['oracle', 'static', 'history', 'lookback21'])
            ax[i].set_title('warehouse {}'.format(i+1))
            ax[i].set_ylabel('stock levels')
            ax[i].set_xlabel('time(days)')
        savepath = os.path.join("output_multilevel", env_name,'stock level.jpg')
        fig.savefig(savepath)