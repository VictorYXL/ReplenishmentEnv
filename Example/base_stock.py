import argparse
import os
import sys
import gym
import numpy as np
import cvxpy as cp

env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)

from ReplenishmentEnv import make_env

def get_single_stock_level(
    selling_price: np.array,
    procurement_cost: np.array,
    demand: np.array,
    vlt: int,
    in_stock: int,
    holding_cost: int ,
    replenishment_before: list = None
) -> np.ndarray:
        # time_hrz_len = history_len + 1 + future_len
        time_hrz_len = len(selling_price)

        # Inventory on hand.
        stocks = cp.Variable(time_hrz_len + 1, integer=True)
        # Inventory on the pipeline.
        transits = cp.Variable(time_hrz_len + 1, integer=True)
        sales = cp.Variable(time_hrz_len, integer=True)
        buy = cp.Variable(time_hrz_len + vlt, integer=True)
        # Requested product quantity from upstream.
        buy_in = cp.Variable(time_hrz_len, integer=True)
        # Expected accepted product quantity.
        buy_arv = cp.Variable(time_hrz_len, integer=True)
        # stock_level = cp.Variable(time_hrz_len, integer=True)
        stock_level = cp.Variable(1, integer=True)

        profit = cp.Variable(1)
        # Add constraints.
        constraints = [
            # Variable lower bound.
            stocks >= 0,
            transits >= 0,
            sales >= 0,
            buy >= 0,
            # Initial values.
            stocks[0] == in_stock,
            # Recursion formulas.
            stocks[1 : time_hrz_len + 1] == stocks[0:time_hrz_len] + buy_arv - sales,
            transits[1 : time_hrz_len + 1] == transits[0:time_hrz_len] - buy_arv + buy_in,
            sales <= stocks[0:time_hrz_len],
            sales <= demand,
            buy_in == buy[vlt : time_hrz_len + vlt],
            buy_arv == buy[0:time_hrz_len],
            stock_level == stocks[0:time_hrz_len] + transits[0:time_hrz_len] + buy_in,
            # Objective function.
            profit == cp.sum(
                cp.multiply(selling_price, sales) - cp.multiply(procurement_cost, buy_in) - cp.multiply(holding_cost, stocks[1:]),
            ),
        ]
            # Init the buy before action
        if replenishment_before is not None:
            constraints.append(transits[0] == sum(replenishment_before))
            for i in range(len(replenishment_before)):
                constraints.append(buy[i] == replenishment_before[i])

        obj = cp.Maximize(profit)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GLPK_MI, verbose=False)
        # return stock_level.value
        return stock_level.value

def get_stock_level(env: gym.Wrapper, env_mode = "test"):
    stock_levels = np.zeros((len(env.get_warehouse_list()), len(env.get_sku_list())))
    for warehouse_index, warehouse in enumerate(env.get_warehouse_list()):
        for sku_index, sku in enumerate(env.get_sku_list()):
            selling_price        = env.get_selling_price(warehouse, sku)
            procurement_cost     = env.get_procurement_cost(warehouse, sku)
            demand               = env.get_demand(warehouse, sku)
            average_vlt          = env.get_average_vlt(warehouse, sku)
            in_stock             = env.get_in_stock(warehouse, sku)
            holding_cost         = env.get_holding_cost(warehouse, sku)
            replenishment_before = env.get_replenishment_before(warehouse, sku)
            # train mode doesn't have replenishment before
            if env_mode =="train":
                stock_level = get_single_stock_level(
                    selling_price, 
                    procurement_cost, 
                    demand, 
                    average_vlt, 
                    in_stock, 
                    holding_cost
                ).reshape(-1, 1)
            else:
                stock_level = get_single_stock_level(
                    selling_price, 
                    procurement_cost, 
                    demand, 
                    average_vlt, 
                    in_stock, 
                    holding_cost,
                    replenishment_before
                ).reshape(-1, 1)
            stock_levels[warehouse_index, sku_index] = stock_level
    return stock_levels

def base_stock(env: gym.Wrapper, update_freq =7, static_stock_levels = None):
    env.reset()
    current_step = 0
    is_done = False
    while not is_done:
        if current_step % update_freq == 0:
            if isinstance(static_stock_levels,np.ndarray):
                stock_levels = static_stock_levels
            else:
                stock_levels = get_stock_level(env)
        replenish = stock_levels - env.get_in_stock() - env.get_in_transit()
        replenish = np.where(replenish >= 0, replenish, 0) / (env.get_demand_mean() + 0.00001)
        states, reward, is_done, info = env.step(replenish)
        current_step += 1
    return info["balance"]


if __name__ == "__main__":
    env_names = [
        "sku50.multi_store.standard",
        # "sku100.multi_store.standard",
        # "sku200.multi_store.standard"
        # "sku500.multi_store.standard",
        # "sku1000.multi_store.standard",
        # "sku2000.multi_store.standard",
    ]

    for env_name in env_names:
        exp_name = "static_base_stock"
        vis_path = os.path.join("output", env_name, exp_name)

        env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="train", vis_path=vis_path)
        env_train.reset()
        env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        static_stock_levels = get_stock_level(env_train, env_mode = "train")
        balance = base_stock(env_test, static_stock_levels = static_stock_levels)
        env_test.render()
        print(env_name, exp_name, balance)

        # env = make_env(env_name, wrapper_names=["StaticWrapper"], mode="test", vis_path=vis_path)
        # balance = base_stock(env)
        # env.render()
        # print(env_name, exp_name, balance)

        # exp_name = "dynamic_base_stock"
        # vis_path = os.path.join("output", env_name, exp_name)
        # env = make_env(env_name, wrapper_names=["DynamicWrapper"], mode="test", vis_path=vis_path)
        # balance = base_stock(env)
        # env.render()
        # print(env_name, exp_name, balance)
