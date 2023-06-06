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

def get_multilevel_stock_level(env: gym.Wrapper):
    stock_levels = np.zeros((len(env.get_warehouse_list()), len(env.get_sku_list())))
    for sku_index, sku in enumerate(env.get_sku_list()):
        selling_price = np.array(env.get_selling_price(sku = sku))
        procurement_cost = np.array(env.get_procurement_cost(sku = sku))
        demand = np.array(env.get_demand(sku = sku))
        average_vlt = env.get_average_vlt(sku = sku).max()
        holding_cost = np.array(env.get_holding_cost(sku = sku))
        if demand.ndim == 2:
            demand = demand[-1].reshape(-1)
        stock_level = get_multilevel_single_stock_level(
                selling_price,
                procurement_cost,
                demand,
                average_vlt,
                holding_cost,
                env.warehouse_count,
            ).reshape(-1,)
        stock_levels[:, sku_index] = stock_level
    return stock_levels

def get_multilevel_single_stock_level(
        selling_price: np.array,
        procurement_cost: np.array,
        demand: np.array,
        vlt: int,
        holding_cost: np.array,
        warehouses: int,
) -> np.ndarray:
    time_hrz_len = len(selling_price[0])
    stocks = cp.Variable((warehouses, time_hrz_len + 1), integer=True)
    transits = cp.Variable((warehouses, time_hrz_len + 1), integer=True)
    sales = cp.Variable((warehouses, time_hrz_len), integer=True)
    buy = cp.Variable((warehouses, time_hrz_len + vlt), integer=True)
    buy_in = cp.Variable((warehouses, time_hrz_len), integer=True)
    buy_arv = cp.Variable((warehouses, time_hrz_len), integer=True)
    stock_level = cp.Variable((warehouses, 1), integer=True)
    profit = cp.Variable(1)
    common_constraints = [
        stocks >= 0,
        transits >= 0,
        sales >= 0,
        buy >= 0
    ]
    intralevel_constraints = [
        stocks[:, 1: time_hrz_len + 1] == stocks[:, 0:time_hrz_len] + buy_arv - sales,
        transits[:, 1: time_hrz_len + 1] == transits[:, 0:time_hrz_len] - buy_arv + buy_in,
        sales <= stocks[:, 0:time_hrz_len],
        buy_in == buy[:, vlt: time_hrz_len + vlt],
        buy_arv == buy[:, 0:time_hrz_len],
        stock_level == stocks[:, 0:time_hrz_len] + transits[:, 0:time_hrz_len] + buy_in,
        transits[:,0] == cp.sum(buy[:,:vlt],axis=1)
    ]
    intralevel_constraints.append(
            profit == cp.sum(
                cp.multiply(selling_price, sales) - cp.multiply(procurement_cost, buy_in) - cp.multiply(holding_cost,stocks[:, 1:]),
            ) - cp.sum(cp.multiply(procurement_cost[:, 0], transits[:, 0])) - cp.sum(cp.multiply(procurement_cost[:, 0], stocks[:,0]))
        )
    interlevel_constraints = []
    for i in range(warehouses):
        if i != warehouses-1:
            interlevel_constraints.append(sales[i] == buy_in[i+1])
        else:
            interlevel_constraints.append(sales[i] <= demand)
    
    constraints = common_constraints + intralevel_constraints + interlevel_constraints
    obj = cp.Maximize(profit)
    prob = cp.Problem(obj, constraints)

    prob.solve(verbose=False)
    if prob.status != 'optimal':
        prob.solve(solver=cp.GLPK_MI, verbose=False, max_iters = 1000)
    if prob.status != 'optimal':
        assert prob.status == 'optimal', 'can\'t find optimal solution for SKU stock level'
    return stock_level.value

def multilevel_base_stock(env: gym.Wrapper, update_freq =7, static_stock_levels = None):
    env.reset()
    current_step = 0
    is_done = False
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((env.warehouse_count, sku_count))
    stock_level_list = [[] for i in range(len(env.get_warehouse_list()))]
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
        current_step += 1

    return info["balance"]

def BS_static(env_name, vis_path):
    exp_name = "BS_static"
    env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="train", vis_path=vis_path)
    env_train.reset()
    static_stock_levels = get_multilevel_stock_level(env_train)
    env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
    balance = multilevel_base_stock(env_test, static_stock_levels = static_stock_levels)
    env_test.render()
    return balance

def BS_dynamic(env_name, vis_path):
    exp_name = "BS_dynamic"
    env = make_env(env_name, wrapper_names=["HistoryWrapper"], mode="test", vis_path=vis_path)
    balance = multilevel_base_stock(env)
    env.render()
    return balance