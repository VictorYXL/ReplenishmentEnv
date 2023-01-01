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
    replenishment_before: list,
    holding_cost: int 
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
        stock_level = cp.Variable(time_hrz_len, integer=True)
        # stock_level = cp.Variable(1, integer=True)

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
            transits[0] == sum(replenishment_before),
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
        for i in range(len(replenishment_before)):
            constraints.append(buy[i] == replenishment_before[i])
        obj = cp.Maximize(profit)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GLPK_MI, verbose=False)
        return np.average(stock_level.value)

def get_stock_level(env: gym.Wrapper):
    stock_levels = np.zeros((len(env.get_facility_list()), len(env.get_sku_list())))
    for facility_index, facility in enumerate(env.get_facility_list()):
        for sku_index, sku in enumerate(env.get_sku_list()):
            selling_price        = env.get_selling_price(facility, sku)
            procurement_cost     = env.get_procurement_cost(facility, sku)
            demand               = env.get_demand(facility, sku)
            average_vlt          = env.get_average_vlt(facility, sku)
            in_stock             = env.get_in_stock(facility, sku)
            holding_cost         = env.get_holding_cost(facility, sku)
            replenishment_before = env.get_replenishment_before(facility, sku)
            stock_level = get_single_stock_level(
                selling_price, 
                procurement_cost, 
                demand, 
                average_vlt, 
                in_stock, 
                replenishment_before, 
                holding_cost
            ).reshape(-1, 1)
            stock_levels[facility_index, sku_index] = stock_level
    return stock_levels

def dynamic_base_stock(env: gym.Wrapper, update_freq=7):
    env.reset()
    current_step = 0
    is_done = False
    while not is_done:
        if current_step % update_freq == 0:
            stock_levels = get_stock_level(env)
        replenish = stock_levels - env.get_in_stock() - env.get_in_transit()
        replenish = np.where(replenish >= 0, replenish, 0) / (env.get_demand_mean() + 0.00001)
        states, reward, is_done, info = env.step(replenish)
        current_step += 1
    return info["balance"]


if __name__ == "__main__":
    env_name = "sku50.MultiStore.Standard"
    env = make_env(env_name, "DynamicWrapper", "test")
    balance = dynamic_base_stock(env)
    print(balance)