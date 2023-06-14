import os
import pdb
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np
import torch.nn.functional as F
import torch

import wandb
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from utils.timehelper import TimeStat
# sys.path.append(os.path.join(os.getcwd(), 'env/ReplenishmentEnv/OR_algorithm'))
# sys.path.append("../..")
# sys.path.append("../../env/ReplenishmentEnv/OR_algorithm/base_stock")
from Baseline.OR_algorithm.base_stock import *
# from ReplenishmentEnv.Baseline.OR_algorithm.base_stock import *
#from env.ReplenishmentEnv.Example.multilevel_base_stock import *
# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunnerWithBasestock:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]

        for i in range(len(env_args)):
            env_args[i]["seed"] += i

        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.n_warehouses = self.env_info["n_warehouses"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []
        self.train_stats = {}
        self.test_stats = {}

        # self.time_stats = defaultdict(lambda: TimeStat(1000))
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac, set_stock_levels = None):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        if not isinstance(set_stock_levels, np.ndarray):
            self.parent_conns[0].send(("get_stock_levels", None))
            self.set_stock_levels = self.parent_conns[0].recv()
        else:
            self.set_stock_levels = set_stock_levels


    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False, storage_capacity=None):

        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
            
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []

        if storage_capacity is not None:
            for parent_conn in self.parent_conns:
                parent_conn.send(("set_storage_capacity", storage_capacity))

    def run(self, lbda_index=None, test_mode=False, 
        visual_outputs_path=None, storage_capacity=None):

        self.reset(test_mode=test_mode, storage_capacity=storage_capacity)

        all_terminated = False
        episode_returns = np.zeros([self.batch_size, self.args.n_lambda])
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_balance = [0 for _ in range(self.batch_size)]
        if self.args.use_n_lambda:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents, self.args.n_lambda])
        else:
            episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents])

        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)

        while True:
            if self.args.mac == "mappo_mac":
                mac_output = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                    bs=envs_not_terminated, test_mode=test_mode)
            elif self.args.mac == "dqn_mac" or self.args.mac == "ldqn_mac":
                mac_output = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, 
                    lbda_indices=None, bs=envs_not_terminated, test_mode=test_mode)
            if save_probs:
                actions, probs = mac_output
            else:
                actions = mac_output
            
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }

            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu").detach()

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step_with_base_stock", (cpu_actions[action_idx], self.set_stock_levels)))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }


            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )
                    post_transition_data["cur_balance"].append(
                        data["info"]["cur_balance"]
                    )
                    
                    episode_returns[idx] += data["reward"]

                    if self.args.n_agents > 1:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"]
                    else:
                        episode_individual_returns[idx] += data["info"]["individual_rewards"][0]

                    episode_balance[idx] = data["info"]["cur_balance"]

                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["mean_action"].append(
                        F.one_hot(actions[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get profit for each env
        episode_profits = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        for parent_conn in self.parent_conns:
            episode_profit = parent_conn.recv()
            episode_profits.append(episode_profit / self.t * (self.episode_limit))

        # Get stats back for each env
        env_stats = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_profits = self.test_profits if test_mode else self.train_profits

        # log_prefix = "test_" if test_mode else ""
        if test_mode:
            log_prefix = "test" if visual_outputs_path is not None else "val"
        else:
            log_prefix = "train"
        if visual_outputs_path is not None:
            self.parent_conns[0].send(("visualize_render",visual_outputs_path))
            self.parent_conns[0].recv()
        infos = [cur_stats] + final_env_infos

        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        max_in_stock_seq = [d['max_in_stock_sum'] for d in final_env_infos]
        cur_stats['max_in_stock_sum'] = np.mean(max_in_stock_seq)

        mean_in_stock_seq = [d['mean_in_stock_sum'] for d in final_env_infos]
        cur_stats['mean_in_stock_sum'] = np.mean(mean_in_stock_seq)

        for i in range(self.n_warehouses):
            mean_in_stock_store_seq = [d['mean_in_stock_sum_store_'+str(i+1)] for d in final_env_infos]
            mean_excess_store_seq = [d['mean_excess_sum_store_'+str(i+1)] for d in final_env_infos]
            mean_backlog_store_seq = [d['mean_backlog_sum_store_'+str(i+1)] for d in final_env_infos]
            cur_stats['mean_in_stock_sum_store_'+str(i+1)] = np.mean(mean_in_stock_store_seq)
            cur_stats['mean_excess_sum_store_'+str(i+1)] = np.mean(mean_excess_store_seq)
            cur_stats['mean_backlog_sum_store_'+str(i+1)] = np.mean(mean_backlog_store_seq)
        cur_returns.extend(episode_returns)
        cur_profits.extend(episode_profits)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        cur_profits = np.array(cur_profits)
        if test_mode:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)

            return cur_stats, lambda_return, profits
        else:
            cur_returns = np.array(cur_returns)
            mean_returns = cur_returns.mean(axis=0)
            lambda_return = mean_returns[lbda_index]
            profits = (cur_profits.mean(axis=0)).sum(axis=-1)
            return self.batch, cur_stats, lambda_return, profits

    def get_overall_avg_balance(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        cur_balances = []
        for parent_conn in self.parent_conns:
            cur_balances.append(parent_conn.recv())

        return np.mean(np.sum(np.array(cur_balances), axis=1))

    def _log(self, returns, individual_returns, profits, stats, prefix):
        self.logger.log_stat(prefix + "_return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "_return_std", np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "_profit_mean", np.mean(profits), self.t_env)
        self.logger.log_stat(prefix + "_profit_std", np.std(profits), self.t_env)
        profits.clear()

        if self.args.use_wandb and self.args.n_agents <= 100:
            for i in range(self.args.n_agents):
                wandb.log(
                    {
                        f"SKUReturn/joint_{prefix}_sku{i+1}_mean": individual_returns[
                            :, i
                        ].mean()
                    },
                    step=self.t_env,
                )

            for i in range(self.args.n_agents):
                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_reward_dict", None))
                reward_dicts = []
                for parent_conn in self.parent_conns:
                    reward_dicts.append(parent_conn.recv())

                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_profit", None))
                cur_balances = []
                for parent_conn in self.parent_conns:
                    cur_balances.append(parent_conn.recv())
                wandb.log(
                    {
                        f"SKUReturn_{k}/joint_{prefix}_sku{i+1}_mean": np.mean(
                            [np.array(rd[k])[:, i].sum() / 1e6 for rd in reward_dicts]
                        )
                        for k in reward_dicts[0].keys()
                    },
                    step=self.t_env,
                )
                wandb.log(
                    {
                        f"SKUBalance/joint_{prefix}_sku{i+1}_mean": np.mean(
                            np.array(cur_balances)[:, i]
                        )
                    },
                    step=self.t_env,
                )
            wandb.log(
                    {
                        f"SumBalance/joint_{prefix}_sum": np.mean(
                            np.sum(np.array(cur_balances), 1)
                        )
                    },
                    step=self.t_env,
            )    

        if self.args.use_wandb:
            wandb.log(
                    {
                        f"instock_sum/{prefix}_max_in_stock_sum_mean": stats['max_in_stock_sum_mean'],
                        f"instock_sum/{prefix}_max_in_stock_sum_min": stats['max_in_stock_sum_min'],
                        f"instock_sum/{prefix}_max_in_stock_sum_max": stats['max_in_stock_sum_max'],
                    },
                    step=self.t_env,
            )    
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_mean", stats['max_in_stock_sum_mean'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_min", stats['max_in_stock_sum_min'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_max", stats['max_in_stock_sum_max'], self.t_env
        )

        for k, v in stats.items():
            if k not in ["n_episodes", "individual_rewards"]:
                self.logger.log_stat(
                    prefix + "_" + k + "_mean", v / stats["n_episodes"], self.t_env
                )

        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step_with_base_stock":
            action_from_rl, stock_levels = data
            replenish = stock_levels - env._env.get_in_stock() - env._env.get_in_transit()
            replenish = np.where(replenish >= 0, replenish, 0) / (env._env.get_demand_mean() + 0.00001)

            discrete_action = env._env.config['action']['space']
            actions = np.array([[np.argmin(np.abs(np.array(discrete_action) - a)) for a in row]  
                for row in replenish])
            # RL algorithm with BSs. Top layer uses RL algorithm and others use BSs.
            actions[0] = action_from_rl
            actions = actions.flatten()
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            env_info['cur_balance'] = env_info['cur_balance'].reshape(env.n_warehouses, -1).sum(axis = 0)
            env_info['individual_rewards'] = env_info['individual_rewards'].reshape(env.n_warehouses, -1).sum(axis = 0)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()[:int(env.get_state_size()/env.n_warehouses)]
            avail_actions = env.get_avail_actions()[:int(len(env.get_avail_actions())/env.n_warehouses)]
            obs = env.get_obs()[:int(len(env.get_obs())/env.n_warehouses)]
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state()[:int(env.get_state_size()/env.n_warehouses)],
                    "avail_actions": env.get_avail_actions()[:int(len(env.get_avail_actions())/env.n_warehouses)],
                    "obs": env.get_obs()[:int(len(env.get_obs())/env.n_warehouses)]
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "switch_mode":
            mode = data
            env.switch_mode(mode)
        elif cmd == "get_profit":
            remote.send(env.get_profit().reshape(env.n_warehouses, -1).sum(axis = 0))
        elif cmd == "get_reward_dict":
            remote.send(env._env.reward_monitor)
        elif cmd == "visualize_render":
            env.visualize_render(data)
            # profit = env.get_profit()
            # print("test_cur_avg_balances : {}".format(profit.sum()))
        elif cmd == "get_storage_capacity":
            remote.send(env._env.storage_capacity)
        elif cmd == "set_storage_capacity":
            env.set_storage_capacity(data)
        elif cmd == "get_stock_levels":
            stock_levels = get_multilevel_stock_level(env._env)
            remote.send(stock_levels)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)