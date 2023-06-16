import numpy as np 
import gym
from typing import Tuple
from ReplenishmentEnv.wrapper.default_wrapper import DefaultWrapper


"""
    DynamicWrapper provides the lookback info,
    including oracle selling_price, procurement_cost, demand, vlt and unit_storage_cost.
"""
class DynamicWrapper(DefaultWrapper):
    def __init__(self, env: gym.Env) -> None:
        self.env = env
    
    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        states, rewards, done, infos = self.env.step(actions)
        return states, rewards, done, infos

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update.
    """
    def reset(self) -> None:
        return self.env.reset()
    
    def get_selling_price(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "selling_price", "lookback", sku].copy()
    
    def get_procurement_cost(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "procurement_cost", "lookback", sku].copy()

    def get_sale(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "sale", "lookback", sku].copy()

    def get_demand(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "demand", "lookback", sku].copy()
    
    def get_average_vlt(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        vlts = self.env.agent_states[warehouse, "vlt", "lookback", sku]
        # for convenience, just adopt the max vlt among all warehouses
        if warehouse== "all_warehouses":
            average_vlt = np.average(vlts, 1).astype('int64')
        else:
            average_vlt = int(np.average(vlts, 0))
        return average_vlt
    

    def get_holding_cost(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        if warehouse == "all_warehouses":
            # Convert to np.ndarray type and reshape
            unit_storage_cost = np.array([self.env.supply_chain[warehouse, "unit_storage_cost"] for warehouse in self.env.warehouse_list]).reshape((-1,1))
        else:
            unit_storage_cost = self.env.supply_chain[warehouse, "unit_storage_cost"]
        basic_holding_cost = self.env.agent_states[warehouse, "basic_holding_cost", "lookback", sku]
        volume = self.env.agent_states[warehouse, "volume", "lookback", sku]
        holding_cost = basic_holding_cost + unit_storage_cost * volume
        return holding_cost
    
    def get_in_stock(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "in_stock", "today", sku].copy()

    def get_in_transit(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "in_transit", "today", sku].copy()