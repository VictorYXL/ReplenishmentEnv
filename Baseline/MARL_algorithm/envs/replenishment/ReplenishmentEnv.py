import re
import gym
import numpy as np
import datetime
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
from gym.wrappers import TimeLimit as GymTimeLimit
from ..multiagentenv import MultiAgentEnv
from ReplenishmentEnv import make_env

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self._env = env

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )
    def reset(self):
        return self._env.reset()
    def step(self, actions):
        return self._env.step(actions)

class ReplenishmentEnv(MultiAgentEnv):
    def __init__(
        self,
        # map_name="n50c500d1",
        n_agents = 100,
        task_type = "Standard",
        mode = "train",
        time_limit=1460,
        vis_path=None,
        **kwargs,
    ):  
        action_space = [0.00, 0.16, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.83, 
                        1.00, 1.16, 1.33, 1.50, 1.66, 1.83, 2.00, 2.16, 2.33, 2.50, 
                        2.66, 2.83, 3.00, 3.16, 3.33, 3.50, 3.66, 3.83, 4.00, 5.00, 
                        6.00, 7.00, 9.00, 12.00]
        update_config = {
                            "action" : {"mode": "demand_mean_discrete",
                                        "space": action_space}
                        } 
        env_base = make_env(task_type, wrapper_names = ["ObservationWrapper4OldCode", "FlattenWrapper", "OracleWrapper"], 
                            mode=mode, vis_path=vis_path, update_config=update_config)
        sampler_seq_len = env_base.config['env']['horizon']
        self.episode_limit = min(time_limit, sampler_seq_len)
        self.env_t = 0
        self.stock_levels = None
        env_base.reset()
        self._env = TimeLimit(
            env_base,
            max_episode_steps = sampler_seq_len,
        )
        self._env = FlattenObservation(self._env)
        self.n_warehouses = self._env.n_warehouses
        self.n_agents = self._env.get_agent_count()
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        # self._env.seed(self._seed)
        self.C_trajectory = None

    def step(self, actions):
        """Returns reward, terminated, info"""
        self.env_t += 1
        # actions = actions.reshape(-1)
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        stats = {"individual_rewards": np.array(reward).astype(np.float32)/1e4
                if self.n_agents > 1
                else np.array([reward,]).astype(np.float32)/1e4,
                "cur_balance": info['profit'],
                "max_in_stock_sum": info['max_in_stock_sum'],
                "mean_in_stock_sum": info['mean_in_stock_sum']
                }
        for i in range(self._env.n_warehouses):
            stats['mean_in_stock_sum_store_'+str(i+1)] = info['mean_in_stock_sum_store_'+str(i+1)]
            stats['mean_excess_sum_store_'+str(i+1)] = info['mean_excess_sum_store_'+str(i+1)]
            stats['mean_backlog_sum_store_'+str(i+1)] = info['mean_backlog_sum_store_'+str(i+1)]
        return (
            float(sum(reward))/1e4,
            done, 
            stats
        )

    def get_obs(self):
        """Returns all agent observations in a list"""
        assert not np.isnan(self._obs).any()
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """Returns initial observations and states"""
        self._obs = self._env.reset()
        self.env_t = 0
        # self.stock_levels = None
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        self.C_trajectory = np.empty([self.episode_limit + 1, 3, self.n_agents])
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def switch_mode(self, mode):
        self._env.switch_mode(mode)

    def get_profit(self):
        profit = self._env.per_balance.copy()
        return profit

    def set_C_trajectory(self, C_trajectory):
        self._env.set_C_trajectory(C_trajectory)

    def set_local_SKU(self, local_SKU):
        self._env.set_local_SKU(local_SKU)
        self.n_agents = self._env.get_agent_count()
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

    def get_C_trajectory(self):
        return self.C_trajectory
    
    def visualize_render(self, visual_output_path):
        return self._env.render()
