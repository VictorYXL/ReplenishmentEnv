import numpy as np 
import gym
from typing import Tuple


"""
    OracleWrapper provides the oracle info after warmup 
    including oracle selling_price, procurement_cost, demand, vlt and unit_storage_cost for oracle algorithm.
"""
class OracleWrapper(gym.Wrapper):
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
    def reset(self, update_config:dict = None) -> None:
        self.env.reset(update_config)
    
    def get_selling_price(self, sku="all_agents") -> np.array:
        return self.env.agent_states["selling_price", "all_dates", sku][self.env.lookback_len:].copy()
    
    def get_procurement_cost(self, sku="all_agents") -> np.array:
        return self.env.agent_states["procurement_cost", "all_dates", sku][self.env.lookback_len:].copy()

    def get_sale(self, sku="all_agents") -> np.array:
        return self.env.agent_states["sale", "all_dates", sku][self.env.lookback_len:].copy()

    def get_demand(self, sku="all_agents") -> np.array:
        return self.env.agent_states["demand", "all_dates", sku][self.env.lookback_len:].copy()
    
    def get_average_vlt(self, sku="all_agents") -> np.array:
        vlts = self.env.agent_states["vlt", "all_dates", sku][self.env.lookback_len:]
        return int(np.average(vlts))
    
    def get_in_stock(self, sku="all_agents") -> np.array:
        return self.env.agent_states["all_facilities", "in_stock", "today", sku].copy()

    def get_unit_storage_cost(self, sku="all_agents") -> np.array:
        return self.env.agent_states["storage_cost", "all_dates", sku][self.env.lookback_len:] + \
            self.env.agent_states["holding_cost_ratio", "all_dates", sku][self.env.lookback_len:] * \
            self.env.agent_states["selling_price", "all_dates", sku][self.env.lookback_len:]
    
    def get_replenishment_before(self, sku="all_agents") -> np.array:
        return self.env.agent_states["replenish", "history", sku][-self.get_average_vlt(sku):]