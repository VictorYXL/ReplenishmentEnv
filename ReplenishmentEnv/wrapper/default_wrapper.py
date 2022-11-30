
import numpy as np 
import gym

class DefaultWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env
        
    def step(self, actions: np.array) -> tuple[np.array, np.array, list, dict]:
        states, rewards, done, infos = self.env.step(actions)
        return states, rewards, done, infos

    def reset(self, random_interception:bool = True) -> None:
        return self.env.reset(random_interception)
    
    # get demean mean by last lookback_len days.
    def get_demand_mean(self) -> np.array:
        mean_demand = np.average(self.agent_states["demand", "lookback"], 0)
        return mean_demand
    
    def get_in_stock(self) -> np.array:
        return self.agent_states["in_stock"].copy()

    def get_in_transit(self) -> np.array:
        return self.agent_states["in_transit"].copy()