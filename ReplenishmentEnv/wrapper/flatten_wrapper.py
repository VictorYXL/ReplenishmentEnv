import numpy as np 
import gym
import scipy.stats as st
from gym import spaces
from typing import Tuple

class FlattenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.sku_count = len(self.env.sku_list)
        self.facility_count = self.supply_chain.get_facility_count()
        self.agent_count = self.sku_count * self.facility_count
    
    def reset(self, vis_path:str=None, update_config: dict=None) -> None:
        states = self.env.reset(vis_path, update_config)
        states = states.reshape((self.agent_count, -1))
        return states

    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        actions = np.array(actions).reshape(self.facility_count, self.sku_count)
        states, rewards, done, infos = self.env.step(actions)
        states = states.reshape((self.agent_count, -1))
        rewards = rewards.flatten()
        return states, rewards, done, infos