import numpy as np 
import gym
import scipy.stats as st
from gym import spaces
from typing import Tuple

"""
    ObservationWrapper can generate more information state, which can help the training of RL algorithm. 
    And ObservationWrapper can be used for old algo code https://github.com/songCNMS/replenishment-marl-baselines.
"""
class ObservationWrapper4OldCode(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env
        self.n_skus = len(self.env.sku_list)
        self.n_warehouses = self.supply_chain.get_warehouse_count()
        self.n_agents = self.n_skus * self.n_warehouses

        self.column_lens = dict()
        self.column_lens["demand_hist"] = self.env.lookback_len
        self.column_lens["hist_order"] = self.env.lookback_len 
        self.column_lens["intransit_hist_sum"] = self.env.lookback_len
        
        self.state_dim = 0
        self.state_info = ["local_info", "global_info", "level_info"]
        if "local_info" in self.state_info:
            self.local_info_dim = (16 + self.column_lens["demand_hist"] + self.column_lens["hist_order"])
            self.state_dim += self.local_info_dim
        if "mean_field" in self.state_info:
            self.state_dim += self.local_info_dim
        if "global_info" in self.state_info:
            self.state_dim += (7 + self.column_lens["intransit_hist_sum"])
        if "rank_info" in self.state_info:
            self.state_dim += 7
        if "level_info" in self.state_info:
            self.state_dim += self.n_warehouses

        self.state = np.zeros((len(self.env.sku_list), self.state_dim))

        # Modify action_space and observation_space to match old code.
        self.agent_observation_space = spaces.Box(
            low=-5000.00, high=5000.00, shape=(self.state_dim,), dtype=np.float64
        )
        self.observation_space = [self.agent_observation_space] * len(self.env.sku_list)
        self.agent_action_space = spaces.Discrete(34)
        self.action_space = [self.agent_action_space] * self.n_agents

        # add mode to match old code.
        self.mode = "train"
        self.in_stock_sum = []
        self.in_stock = []
        self.excess_sum = []
        self.backlog_sum = []

        
    """
        Generate the state of all skus from all warehouses, called state_v1.
        Returns:
        state_v1[np.array] : (n_warehouses * n_skus, n_dim) The state of all skus, including global information
         and local information.
    """
    def get_state_v1(self) -> np.array:
        demand_mean = np.average(self.env.agent_states["all_warehouses", "demand", "lookback_with_current"], 1)

        state_normalize = demand_mean + 1
        state_normalize_reshape = state_normalize[:, :, np.newaxis]
        price_normalize = self.env.agent_states["all_warehouses", "selling_price"]

        state_list = []
        storage_capacity = np.array([self.supply_chain[warehouse, "capacity"] for warehouse in self.supply_chain.get_warehouse_list()])[:, np.newaxis]
        unit_storage_cost = np.array([self.supply_chain[warehouse, "unit_storage_cost"] for warehouse in self.supply_chain.get_warehouse_list()])[:, np.newaxis]
        if "local_info" in self.state_info:
            # is_out_of_stock
            state_list.append(np.where(self.env.agent_states["all_warehouses", "in_stock"] <= 0, 1.0, 0.0)[:, :, np.newaxis])

            # inventory_in_stock
            state_list.append((self.env.agent_states["all_warehouses", "in_stock"] / state_normalize)[:, :, np.newaxis])

            # inventory_in_transit
            state_list.append((self.env.agent_states["all_warehouses", "in_transit"] / state_normalize)[:, :, np.newaxis])

            # inventory_estimated
            inventory_estimated = self.env.agent_states["all_warehouses", "in_stock"] + self.env.agent_states["all_warehouses", "in_transit"]
            state_list.append((inventory_estimated / state_normalize)[:, :, np.newaxis])

            # inventory_rop
            sale_mean = np.mean(self.env.agent_states["all_warehouses", "sale", "lookback_with_current"], axis=1)
            sale_std = np.std(self.env.agent_states["all_warehouses", "sale", "lookback_with_current"], axis=1)
            inventory_rop = (
                self.env.agent_states["all_warehouses", "vlt"] * sale_mean
                + np.sqrt(self.env.agent_states["all_warehouses", "vlt"])
                * sale_std
                * st.norm.ppf(0.95) # service_levels
            )
            state_list.append((inventory_rop / state_normalize)[:, :, np.newaxis])

            # is_below_rop
            state_list.append(np.where(inventory_estimated <= inventory_rop, 1.0, 0.0)[:, :, np.newaxis])

            # demand_std
            state_list.append((np.std(self.env.agent_states["all_warehouses", "demand", "lookback_with_current"], axis=1) / state_normalize)[:, :, np.newaxis])

            # demand_hist
            state_list.append(self.env.agent_states["all_warehouses", "demand", "lookback_with_current"].transpose([0, 2, 1]) / state_normalize_reshape)
            # capacity
            state_list.append((demand_mean / storage_capacity)[:, :, np.newaxis])

            # sku_price
            state_list.append((self.env.agent_states["all_warehouses", "selling_price"] / price_normalize)[:, :, np.newaxis])

            # sku_cost
            state_list.append((self.env.agent_states["all_warehouses", "procurement_cost"] / price_normalize)[:, :, np.newaxis])
            
            # sku_profit
            sku_profit = self.env.agent_states["all_warehouses", "selling_price"] - self.env.agent_states["all_warehouses", "procurement_cost"]
            state_list.append((sku_profit / price_normalize)[:, :, np.newaxis])

            # holding_cost
            holding_cost = (self.env.agent_states["all_warehouses", "basic_holding_cost"] + 
                        unit_storage_cost * self.env.agent_states["all_warehouses", "volume"])
            state_list.append((holding_cost / price_normalize)[:, :, np.newaxis])

            # order_cost
            state_list.append((self.env.agent_states["all_warehouses", "unit_order_cost"] / price_normalize)[:, :, np.newaxis])

            # vlt
            state_list.append(self.env.agent_states["all_warehouses", "vlt"][:, :, np.newaxis])

            # vlt_demand_mean
            state_list.append(((demand_mean * (self.env.agent_states["all_warehouses", "vlt"] + 1)) / state_normalize)[:, :, np.newaxis])

            # vlt_day_remain
            state_list.append(((inventory_estimated - demand_mean * 
                                (self.env.agent_states["all_warehouses", "vlt"] + 1)) / state_normalize)[:, :, np.newaxis])

            # hist_order
            state_list.append(self.env.agent_states["all_warehouses", "replenish", "lookback_with_current"].transpose([0,2,1]) / \
                            state_normalize_reshape)

        if "global_info" in self.state_info:
            # in_stock_sum
            # state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum(self.env.agent_states["all_warehouses", "in_stock"], axis=-1) / storage_capacity)[:, :, np.newaxis])
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum(self.env.agent_states["all_warehouses", "in_stock"], axis=-1)[:, np.newaxis] / storage_capacity)[:, :, np.newaxis])

            # in_stock_profit
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum(self.env.agent_states["all_warehouses", "in_stock"] * sku_profit, axis=-1)[:, np.newaxis] / \
                        ((np.sum(self.env.agent_states["all_warehouses", "in_stock"], axis=-1) + 1)[:, np.newaxis] * price_normalize))[:, :, np.newaxis])

            # remain_capacity
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum(storage_capacity - self.env.agent_states["all_warehouses", "in_stock"], axis=-1)[:, np.newaxis] /
                                            storage_capacity)[:, :, np.newaxis])

            # intransit_sum
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum(self.env.agent_states["all_warehouses", "in_transit"], axis=-1)[:, np.newaxis] / storage_capacity)[:, :, np.newaxis])

            # intransit_hist_sum
            state_list.append(self.env.agent_states["all_warehouses", "replenish", "lookback_with_current"].transpose([0,2,1]) / \
                            storage_capacity[:, :, np.newaxis])

            # intransit_profit
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * (self.env.agent_states["all_warehouses", "in_transit"] * sku_profit).sum() / \
                        ((np.sum(self.env.agent_states["all_warehouses", "in_transit"], axis=-1)[:, np.newaxis] + 1) * price_normalize))[:, :, np.newaxis])

            # instock_intransit_sum
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum(inventory_estimated, axis=-1)[:, np.newaxis] / storage_capacity)[:, :, np.newaxis])

            # instock_intransit_profit
            state_list.append((np.ones((self.n_warehouses, self.n_skus)) * np.sum((inventory_estimated * sku_profit), axis=-1)[:, np.newaxis] / \
                        ((np.sum(inventory_estimated, axis=-1)[:, np.newaxis] + 1) * price_normalize))[:, :, np.newaxis])
            
        if "level_info" in self.state_info:
            one_hot = np.repeat(np.eye(self.n_warehouses), self.n_skus, axis = 0).reshape((self.n_warehouses, self.n_skus, self.n_warehouses))
            state_list.append(one_hot)

        # Rank infos
        if "rank_info" in self.state_info:
            state_list.append((self.env.agent_states["all_warehouses", "in_stock"].argsort() / self.n_skus)[:, :, np.newaxis])
            state_list.append((self.env.agent_states["all_warehouses", "in_transit"].argsort() / self.n_skus)[:, :, np.newaxis])
            state_list.append((inventory_estimated.argsort() / self.n_skus)[:, :, np.newaxis])
            state_list.append((demand_mean.argsort() / self.n_skus)[:, :, np.newaxis])
            state_list.append((sku_profit.argsort() / self.n_skus)[:, :, np.newaxis])
            state_list.append((self.env.agent_states["all_warehouses", "selling_price"].argsort() / self.n_skus)[:, :, np.newaxis])
            state_list.append((self.env.agent_states["all_warehouses", "procurement_cost"].argsort() / self.n_skus)[:, :, np.newaxis])

        state = np.concatenate(state_list, axis = -1)

        if "mean_field" in self.state_info:
            mean_info = state[:, :self.local_info_dim].mean(axis = 0, keepdims = True)
            mean_info = np.tile(mean_info, (self.n_skus, 1))
            state = np.concatenate([state, mean_info], axis = -1)
        # state = state.reshape(self.n_skus * self.n_warehouses, -1)
        # state = np.nan_to_num(state)
        return state

    """
        Step orders: Replenish -> Sell -> Receive arrived skus -> Update balance
        actions: [action_idx/action_quantity] by sku order, defined by action_setting in config
        Returns:
        state_v1[np.array] : (n_skus * n_warehouses, n_dim) The state of all skus, including global information
         and local information.
    """
    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        _, rewards, done, infos = self.env.step(actions)
        self.env.pre_step()
        states = self.get_state_v1()
        self.in_stock_sum.append(self.env.agent_states["all_warehouses", "in_stock"].sum())
        excess_cost = self.env.agent_states["all_warehouses","excess"]*self.env.agent_states["all_warehouses","overflow_cost_ratio"]
        backlog_cost = (self.env.agent_states["all_warehouses","selling_price"] - self.env.agent_states["all_warehouses","procurement_cost"]) * (
            self.env.agent_states["all_warehouses","demand"] - self.env.agent_states["all_warehouses","sale"]) * self.env.agent_states["all_warehouses","backlog_ratio"] 
        if not isinstance(self.excess_sum, np.ndarray):
            self.in_stock = self.env.agent_states["all_warehouses", "in_stock"].sum(axis = 1, keepdims = True)
            self.excess_sum = excess_cost.sum(axis = 1, keepdims = True)
            self.backlog_sum = backlog_cost.sum(axis = 1, keepdims = True)
        else:
            self.in_stock = np.concatenate((self.in_stock, self.env.agent_states["all_warehouses", "in_stock"].sum(axis = 1, keepdims = True)), axis = 1)
            self.excess_sum = np.concatenate((self.excess_sum, excess_cost.sum(axis = 1, keepdims = True)), axis = 1)
            self.backlog_sum = np.concatenate((self.backlog_sum, backlog_cost.sum(axis = 1, keepdims = True)), axis = 1)
        self.env.next_step()

        infos['cur_balance'] = self.env.per_balance.copy()
        infos['max_in_stock_sum'] = max(self.in_stock_sum)
        infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum)
        for i in range(self.n_warehouses):
            infos['mean_in_stock_sum_store_'+str(i+1)] = np.mean(self.in_stock[i])
            infos['mean_excess_sum_store_'+str(i+1)] = np.mean(self.excess_sum[i])
            infos['mean_backlog_sum_store_'+str(i+1)] = np.mean(self.backlog_sum[i])
        return states, rewards, done, infos

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update
    """
    def reset(self) -> None:
        self.in_stock_sum = []
        self.excess_sum = []
        self.backlog_sum = []
        self.in_stock = []
        self.env.reset()
        self.env.pre_step()
        states = self.get_state_v1()
        self.env.next_step()
        return states

    """
        switch mode(train/test) for old code.
    """
    def switch_mode(self, mode: str) -> None:
        self.mode = mode
    
    """
        get profit for old code.
    """
    def get_profit(self) -> int:
        return self.env.balance
    
    def get_agent_count(self) -> int:
        return self.n_agents
