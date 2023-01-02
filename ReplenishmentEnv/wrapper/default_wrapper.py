
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
    def reset(self, exp_name = None, update_config:dict = None) -> None:
        self.env.reset(exp_name, update_config)
   
    # get demean mean by last lookback_len days.
    def get_demand_mean(self, facility="all_facilities", sku="all_skus") -> np.array:
        demand = self.agent_states["all_facilities", "demand", "lookback"]
        if facility == "all_facilities":
            mean_demand = np.average(demand, 1)
        else:
            mean_demand = np.average(demand, 0)
        return mean_demand
    
    def get_in_stock(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.agent_states[facility, "in_stock", "today", sku].copy()

    def get_in_transit(self, facility="all_facilities", sku="all_skus") -> np.array:
        return self.agent_states[facility, "in_transit", "today", sku].copy()

    def get_sku_list(self) -> list:
        return self.env.sku_list.copy()
    
    def get_facility_list(self) -> list:
        return self.env.facility_list.copy()