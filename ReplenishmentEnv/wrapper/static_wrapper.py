import numpy as np 
import gym
from typing import Tuple
from .default_wrapper import DefaultWrapper


"""
    StaticWrapper provides the history info,
    including oracle selling_price, procurement_cost, demand, vlt and unit_storage_cost.
"""
class StaticWrapper(DefaultWrapper):
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
    
    def get_selling_price(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.env.agent_states[facility, "selling_price", "history", sku].copy()
    
    def get_procurement_cost(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.env.agent_states[facility, "procurement_cost", "history", sku].copy()

    def get_sale(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.env.agent_states[facility, "sale", "history", sku].copy()

    def get_demand(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.env.agent_states[facility, "demand", "history", sku].copy()
    
    def get_average_vlt(self, facility="all_facilities", sku="all_skus") -> np.array:
        vlts = self.env.agent_states[facility, "vlt", "history", sku]
        if facility== "all_facilities":
            average_vlt = int(np.average(vlts, 1))
        else:
            average_vlt = int(np.average(vlts, 0))
        return average_vlt
    

    def get_holding_cost(self, facility="all_facilities", sku="all_skus") -> np.array:
        if facility == "all_facilities":
            unit_storage_cost = [self.env.supply_chain[facility, "unit_storage_cost"] for facility in self.env.facility_list]
        else:
            unit_storage_cost = self.env.supply_chain[facility, "unit_storage_cost"]
        basic_holding_cost = self.env.agent_states[facility, "basic_holding_cost", "history", sku]
        volume = self.env.agent_states[facility, "volume", "history", sku]
        holding_cost = basic_holding_cost + unit_storage_cost * volume
        return holding_cost
    
    def get_replenishment_before(self, facility="all_facilities", sku="all_skus") -> np.array:
        replenishment = self.env.agent_states[facility, "replenish", "history", sku]
        vlt = self.get_average_vlt(facility, sku)
        replenishment_before = replenishment[-self.lookback_len-vlt:-self.lookback_len]
        return replenishment_before
    
    def get_in_stock(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.env.agent_states[facility, "in_stock", "today", sku].copy()

    def get_in_transit(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.env.agent_states[facility, "in_transit", "today", sku].copy()