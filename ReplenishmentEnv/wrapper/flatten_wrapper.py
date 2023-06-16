import numpy as np 
import gym
from typing import Tuple

class FlattenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.sku_count = len(self.env.sku_list)
        self.warehouse_count = self.supply_chain.get_warehouse_count()
        self.agent_count = self.sku_count * self.warehouse_count
    
    def reset(self) -> None:
        states = self.env.reset()
        states = states.reshape((self.agent_count, -1))
        return states

    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        actions = np.array(actions).reshape(self.warehouse_count, self.sku_count)
        states, rewards, done, infos = self.env.step(actions)
        states = states.reshape((self.agent_count, -1))
        rewards = rewards.flatten()
        infos['profit'] = infos['profit'].flatten()
        return states, rewards, done, infos