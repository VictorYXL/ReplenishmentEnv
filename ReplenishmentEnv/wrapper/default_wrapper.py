
import numpy as np 
import gym
from typing import Tuple

class DefaultWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env
        
    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        states, rewards, done, infos = self.env.step(actions)
        return states, rewards, done, infos

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update
    """
    def reset(self) -> None:
        return self.env.reset()
   
    # get demean mean by last lookback_len days.
    def get_demand_mean(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        demand = self.agent_states["all_warehouses", "demand", "lookback"]
        if warehouse == "all_warehouses":
            mean_demand = np.average(demand, 1)
        else:
            mean_demand = np.average(demand, 0)
        return mean_demand
    
    def get_in_stock(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.agent_states[warehouse, "in_stock", "today", sku].copy()

    def get_in_transit(self, warehouse="all_warehouses", sku="all_skus") -> np.array:
        return self.agent_states[warehouse, "in_transit", "today", sku].copy()

    def get_sku_list(self) -> list:
        return self.env.sku_list.copy()
    
    def get_warehouse_list(self) -> list:
        return self.env.warehouse_list.copy()