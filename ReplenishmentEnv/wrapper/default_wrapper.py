
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
    def reset(self, update_config:dict = None) -> None:
        return self.env.reset(update_config)
   
    # get demean mean by last lookback_len days.
    def get_demand_mean(self) -> np.array:
        mean_demand = np.average(self.agent_states["all_facilities", "demand", "lookback"], 0)
        return mean_demand
    
    def get_in_stock(self) -> np.array:
        return self.agent_states["all_facilities", "in_stock"].copy()

    def get_in_transit(self) -> np.array:
        return self.agent_states["all_facilities", "in_transit"].copy()

    def get_sku_list(self) -> np.array:
        return self.env.sku_list.copy()