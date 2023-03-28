import argparse
import os
import sys
import gym
import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt

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
        # TODO:如果不带上replenishment before，那么transit[0]就会是一个很大的数字！导致了从一开始就是满仓状态，因此受益才会很高
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
                    # TODO:测试如果dynamic不加限制会怎样
                    # replenishment_before
                ).reshape(-1, 1)
            stock_levels[warehouse_index, sku_index] = stock_level
    return stock_levels

def get_multilevel_stock_level(env: gym.Wrapper, env_mode = "test"):
    stock_levels = np.zeros((len(env.get_warehouse_list()), len(env.get_sku_list())))
    for sku_index, sku in enumerate(env.get_sku_list()):
        #selling_price, procurement_cost, demand, average_vlt, in_stock, holding_cost, replenishment_before = [], [], [], [], [], [], []
        selling_price = np.array(env.get_selling_price(sku = sku))
        procurement_cost = np.array(env.get_procurement_cost(sku = sku))
        demand = np.array(env.get_demand(sku = sku))[-1]
        average_vlt = np.array(env.get_average_vlt(sku = sku))
        in_stock = np.array(env.get_in_stock(sku = sku))
        holding_cost = np.array(env.get_holding_cost(sku = sku))
        replenishment_before = np.array(env.get_replenishment_before(sku = sku))
        if env_mode == "train":
            stock_level = get_single_stock_level2(
                selling_price,
                procurement_cost,
                demand,
                average_vlt,
                in_stock,
                holding_cost,
                len(env.get_warehouse_list()),
                replenishment_before
            ).reshape(-1,)
        else:
            stock_level = get_single_stock_level2(
                selling_price,
                procurement_cost,
                demand,
                average_vlt,
                in_stock,
                holding_cost,
                len(env.get_warehouse_list())
            ).reshape(-1,)
        stock_levels[:, sku_index] = stock_level
    return stock_levels

def get_multilevel_single_stock_level(
        #TODO: 这里全部变list了，list里面才是array
        selling_price: np.array,
        procurement_cost: np.array,
        # TODO:这里的demand应该是一个，只包含最底层的需求的一个array
        demand: np.array,
        vlt: int,
        in_stock: np.array,
        holding_cost: np.array,
        warehouses: int,
        replenishment_before: np.array = None
) -> np.ndarray:
    if isinstance(replenishment_before, np.ndarray):
        len_replenishment_before = replenishment_before.shape[-1]
    # time_hrz_len = history_len + 1 + future_len
    time_hrz_len = len(selling_price[0])
    # Inventory on hand.
    #stocks = cp.Variable(time_hrz_len + 1, integer=True)
    stocks = cp.Variable([warehouses, time_hrz_len + 1], integer=True)
    # Inventory on the pipeline.
    # transits = cp.Variable(time_hrz_len + 1, integer=True)
    transits = cp.Variable([warehouses, time_hrz_len + 1], integer=True)
    # sales = cp.Variable(time_hrz_len, integer=True)
    sales = cp.Variable([warehouses, time_hrz_len], integer=True)
    # buy = cp.Variable(time_hrz_len + vlt, integer=True)
    buy = cp.Variable([warehouses, time_hrz_len + vlt], integer=True)
    # Requested product quantity from upstream.
    # buy_in = cp.Variable(time_hrz_len, integer=True)
    buy_in = cp.Variable([warehouses, time_hrz_len], integer=True)
    # Expected accepted product quantity.
    # buy_arv = cp.Variable(time_hrz_len, integer=True)
    buy_arv = cp.Variable([warehouses, time_hrz_len], integer=True)
    # stock_level = cp.Variable(time_hrz_len, integer=True)
    # stock_level = cp.Variable(1, integer=True)
    stock_level = cp.Variable([warehouses, 1], integer=True)

    # profit = cp.Variable(1)
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
        # sales <= stock
        sales <= stocks[:, 0:time_hrz_len],
        # buy_in 相当于论文里的R_t^m，是真正购买的sku的量
        buy_in == buy[:, vlt: time_hrz_len + vlt],
        # 此处是假设每个period内leading time为固定的
        buy_arv == buy[:, 0:time_hrz_len],
        # 即每个时刻都需要补货至stock level
        stock_level == stocks[:, 0:time_hrz_len] + transits[:, 0:time_hrz_len] + buy_in,
        profit == cp.sum(
            cp.multiply(selling_price, sales) - cp.multiply(procurement_cost, buy_in) - cp.multiply(holding_cost,
                                                                                                    stocks[:, 1:]),
        ),
    ]
    if isinstance(replenishment_before, np.ndarray):
        intralevel_constraints.append(transits[:, 0] == np.sum(replenishment_before, axis = 1))
        if len_replenishment_before>0:
            intralevel_constraints.append(buy[:, :len_replenishment_before] == replenishment_before)
    # 这个主要是约束各个层次间demand和sale的关系
    interlevel_constraints = []
    for i in range(warehouses):
        if i != warehouses-1:
            # buy_in == sales
            interlevel_constraints.append(sales[i] == buy_in[i-1])
        else:
            # buy_in == consumer demand
            interlevel_constraints.append(sales[i] <= demand)
    
    constraints = common_constraints + intralevel_constraints + interlevel_constraints
    obj = cp.Maximize(profit)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.GLPK_MI, verbose=False)
    # return stock_level.value
    return stock_level.value

def base_stock(env: gym.Wrapper, update_freq =7, static_stock_levels = None):
    env.reset()
    current_step = 0
    is_done = False
    stock_level_list = [[] for i in range(len(env.get_warehouse_list()))]
    while not is_done:
        if current_step % update_freq == 0:
            if isinstance(static_stock_levels,np.ndarray):
                stock_levels = static_stock_levels
            else:
                stock_levels = get_stock_level(env)
        for i in range(len(env.get_warehouse_list())):
            stock_level_list[i].append(stock_levels[i].sum())

        replenish = stock_levels - env.get_in_stock() - env.get_in_transit()
        replenish = np.where(replenish >= 0, replenish, 0) / (env.get_demand_mean() + 0.00001)
        states, reward, is_done, info = env.step(replenish)
        current_step += 1
    return info["balance"], stock_level_list


if __name__ == "__main__":
    env_names = [
        # "sku50.multi_store.standard",
        # "sku50.multi_store.low_capacity"
        # "sku100.multi_store.standard",
        # "sku200.multi_store.standard"
        # "sku500.multi_store.standard",
        # "sku1000.multi_store.standard",
        # "sku2000.multi_store.standard",
        "sku2.single_store.standard",
    ]

    for env_name in env_names:
        

        # exp_name = "static_base_stock"
        # vis_path = os.path.join("output_multilevel", env_name, exp_name)
        # env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        # env_train.reset()
        # static_stock_levels = get_stock_level(env_train, env_mode = "train")
        # print(static_stock_levels)
        # env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        # env_test.reset()
        # balance, static_stock_levels_list = base_stock(env_test, static_stock_levels = static_stock_levels)
        # env_test.render()
        # # print(env_name, exp_name, balance)
        # profit_list = []
        # lim = 20
        # for i in range(-lim,lim):
        #     print(i)
        #     temp_level = (static_stock_levels*(1+0.01*i)).astype(int)
        #     env_test.reset()
        #     balance, _ = base_stock(env_test, static_stock_levels = temp_level)
        #     profit_list.append(balance)
        # print(profit_list)
        # x = np.arange(-lim,lim)
        # plt.plot(x,profit_list)
        # plt.xlabel('adjust(%)')
        # plt.ylabel('profit')
        # savepath = os.path.join("output_multilevel", 'base stock profit.jpg')
        # plt.savefig(savepath)

        # static_stock_levels = get_stock_level(env_train)
        # env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        # balance, static_stock_levels_list = base_stock(env_test, static_stock_levels = static_stock_levels)
        # env_test.render()F
        # print(env_name, exp_name, balance)
        

        # exp_name = "static"
        # vis_path = os.path.join("output_multilevel", env_name, exp_name)
        # env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="train", vis_path=vis_path)
        # env_train.reset()
        # static_stock_levels = get_stock_level(env_train, env_mode='train')
        # env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
        # balance, static_stock_levels_list= base_stock(env_test, static_stock_levels = static_stock_levels)
        # env_test.render()
        # print(env_name, exp_name, balance)

        exp_name = "static_base_stock"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        stock_levels_list = []
        balance_list = []
        mean_stock_levels_list = []
        for i in range(1,61):
            print(i)
            env_train = make_env(env_name, wrapper_names=["OracleWrapper"], mode="train", vis_path=vis_path)
            env_train.reset()
            static_stock_levels = get_stock_level(env_train, env_mode = "train")
            stock_levels_list.append(static_stock_levels)
            if i%5 == 0:
                print(stock_levels_list)   
                stock_levels_mean = np.mean(np.array(stock_levels_list),axis = 0).astype(int)
                mean_stock_levels_list.append(stock_levels_mean.sum())
                #print(stock_levels_mean)
                env_test = make_env(env_name, wrapper_names=["OracleWrapper"], mode="test", vis_path=vis_path)
                balance, static_stock_levels_list= base_stock(env_test, static_stock_levels = stock_levels_mean)
                env_test.render()
                print(env_name, exp_name, balance)
                balance_list.append(balance[0])
        plt.plot(balance_list)
        plt.xlabel('training times')
        plt.ylabel('profit')
        savepath = os.path.join("output_multilevel", env_name)
        plt.savefig('profit along training times.jpg')

        plt.clf()
        stock_levels_list = [np.sum(x) for x in stock_levels_list]
        plt.plot(stock_levels_list)
        plt.xlabel('training times')
        plt.ylabel('stock levels')
        savepath = os.path.join("output_multilevel", env_name)
        plt.savefig('stock levels along training times.jpg')

        plt.clf()
        plt.plot(mean_stock_levels_list)
        plt.xlabel('training times')
        plt.ylabel('mean stock levels')
        savepath = os.path.join("output_multilevel", env_name)
        plt.savefig('mean stock levels along training times.jpg')

        exp_name = "dynamic_history"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["StaticWrapper"], mode="test", vis_path=vis_path)
        balance, history_stock_levels_list= base_stock(env)
        env.render()
        print(env_name, exp_name, balance)

        exp_name = "dynamic_lookback21"
        vis_path = os.path.join("output_multilevel", env_name, exp_name)
        env = make_env(env_name, wrapper_names=["DynamicWrapper"], mode="test", vis_path=vis_path)
        balance, lookback21_stock_levels_list = base_stock(env)
        env.render()

        print(env_name, exp_name, balance)
        fig,ax = plt.subplots(3,1)
        for i in range(len(env.get_warehouse_list())):
            ax[i].plot(static_stock_levels_list[i])
            ax[i].plot(history_stock_levels_list[i])
            ax[i].plot(lookback21_stock_levels_list[i])
            ax[i].legend(['static', 'history', 'lookback21'])
            ax[i].set_title('warehouse {}'.format(i+1))
        savepath = os.path.join("output_multilevel", env_name)
        fig.savefig(savepath+'stock level.jpg')
