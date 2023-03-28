import numpy as np 
import gym
from typing import Tuple
from .default_wrapper import DefaultWrapper


"""
    OracleWrapper provides the all dates info,asdasd as
    including oracle selling_price, procurement_cost, demand, vlt and unit_storage_cost.
"""
class OracleWrapper(DefaultWrapper):
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
        return self.env.agent_states[warehouse, "selling_price", "all_dates", sku].copy()
    
    def get_procurement_cost(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "procurement_cost", "all_dates", sku].copy()

    def get_sale(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "sale", "all_dates", sku].copy()
    
    # Temporarily set demand of all warehouse to be the demand of the most downstream warehouse
    def get_demand(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states["all_warehouses", "demand", "all_dates", sku][-1].copy()
    
    def get_average_vlt(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        vlts = self.env.agent_states[warehouse, "vlt", "all_dates", sku]
        # for convenience, just adopt the max vlt among all warehouses
        if warehouse== "all_warehouses":
            # average_vlt = np.average(vlts, 1).astype('int64')
            average_vlt = np.average(vlts, 1).astype('int64').max()
        else:
            average_vlt = int(np.average(vlts, 0))
        return average_vlt
    

    def get_holding_cost(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        if warehouse == "all_warehouses":
            # Convert to np.ndarray type and reshape
            unit_storage_cost = np.array([self.env.supply_chain[warehouse, "unit_storage_cost"] for warehouse in self.env.warehouse_list]).reshape((-1, 1))
        else:
            unit_storage_cost = self.env.supply_chain[warehouse, "unit_storage_cost"]
        basic_holding_cost = self.env.agent_states[warehouse, "basic_holding_cost", "all_dates", sku]
        volume = self.env.agent_states[warehouse, "volume", "all_dates", sku]
        holding_cost = basic_holding_cost + unit_storage_cost * volume
        return holding_cost
    
    def get_replenishment_before(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        replenishment = self.env.agent_states[warehouse, "replenish", "history", sku]
        # temporarily adopt the largest vlt among all warehouses as the vlt
        vlt = self.get_average_vlt(warehouse, sku)
        if not isinstance(vlt,int):
            vlt = np.max(vlt)
            replenishment_before = replenishment[:, -self.lookback_len-vlt:-self.lookback_len]
        else:
            # get the number of SKUs in pipeline 
            replenishment_before = replenishment[-self.lookback_len-vlt:-self.lookback_len]
        return replenishment_before
    
    def get_in_stock(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "in_stock", "today", sku].copy()

    def get_in_transit(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.env.agent_states[warehouse, "in_transit", "today", sku].copy()