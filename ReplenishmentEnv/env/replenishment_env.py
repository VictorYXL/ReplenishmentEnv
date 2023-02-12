import copy
import os
from datetime import datetime, timedelta
from gym import Env, spaces
from typing import Tuple
import numpy as np
import pandas as pd
import random
import yaml


from .helper_function.rewards import reward1, reward2
from .helper_function.warmup import replenish_by_last_demand
from .helper_function.convertors import continuous, discrete, demand_mean_continuous, demand_mean_discrete
from .helper_function.accept import equal_accept
from .supply_chain import SupplyChain
from .agent_states import AgentStates
from ..utility.utility import deep_update
from ..utility.visualizer import Visualizer
from ..utility.data_loader import DataLoader


class ReplenishmentEnv(Env):
    def __init__(self, config_path, mode="train", vis_path=None, update_config=None):
        # Config loaded from yaml file
        self.config: dict = None

        # All following variables will be refresh in init_env function
        # Sku list from config file
        self.sku_list: list = None

        # Duration in days
        self.durations = 0

        # Look back information length
        self.lookback_len = 0

        # All facilities' sku data are shored in total_data, including 3 types of sku information.
        # self.total_data = [
        # {
        #       "facility_name" : name,
        #       "shared_data"   : shared_data, 
        #       "static_data"   : static_data, 
        #       "dynamic_data"  : dynamic_data
        # },
        #    ...
        #]
        # Shared info saved shared information for all skus and all dates.
        # Stored as dict = {state item: value} 
        # Static info saved special information for each sku but will not changed.
        # Stored as N * M pd.DataFrame (N: agents_count, M: state item count)
        # Dynamic info saved the information which is different between different skus and will change by date.
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.total_data: list = None

        # Step tick.
        self.current_step = 0

        # Env mode including train, validation and test. Each mode has its own dataset.
        self.mode = mode

        # visualization path
        self.vis_path = os.path.join("output", datetime.now().strftime('%Y%m%d_%H%M%S')) if vis_path is None else vis_path
 
        self.load_config(config_path, update_config)
        self.build_supply_chain()

        self.load_data()
        self.init_env()

    def load_config(self, config_path: str, update_config: dict) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
    
        if update_config is not None:
            self.config = deep_update(self.config, update_config)

        assert("env" in self.config)
        assert("mode" in self.config["env"])
        assert("sku_list" in self.config["env"])
        assert("facility" in self.config)
        assert("profit_function" in self.config)
        assert("reward_function" in self.config)
        assert("output_state" in self.config)
        assert("action" in self.config)
        assert("visualization" in self.config)

    """
        Build supply chain
    """
    def build_supply_chain(self) -> None:
        self.supply_chain   = SupplyChain(self.config["facility"])
        self.facility_list  = self.supply_chain.get_facility_list()
        self.facility_to_id = {facility_name: index for index, facility_name in enumerate(self.facility_list)}
        self.id_to_facility = {index: facility_name for index, facility_name in enumerate(self.facility_list)}
    
    """
        Load all facilities' sku data, including facility name, shared data, static data and dynamic data.
        Only load only in __init__ function.
    """
    def load_data(self) -> None:
        self.total_data = []
        data_loader = DataLoader()

        # Convert the sku list from file to list. 
        if isinstance(self.config["env"]["sku_list"], str):
            self.sku_list = data_loader.load_as_list(self.config["env"]["sku_list"])
        else:
            self.sku_list = self.config["env"]["sku_list"]

        for facility_config in self.config["facility"]:
            assert("sku" in facility_config)
            facility_name   = facility_config["name"]
            sku_config      = facility_config["sku"]

            # Load shared info, which is shared for all skus and all dates.
            # Shared info is stored as dict = {state item: value}
            facility_shared_data = sku_config.get("shared_data", {})

            # Load and check static sku info, which is special for each sku but will not changed.
            # Static info is stored as N * M pd.DataFrame (N: agents_count, M: state item count)
            if "static_data" in sku_config:
                facility_static_data = data_loader.load_as_df(sku_config["static_data"])
            else:
                facility_static_data = np.zeros((len(self.sku_list), 0))

            # Load and check demands info, which is different between different skus and will change by date
            # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
            facility_dynamic_data = {}
            for dynamic_data_item in sku_config.get("dynamic_data", {}):
                assert("name" in dynamic_data_item)
                assert("file" in dynamic_data_item)
                item = dynamic_data_item["name"]
                file = dynamic_data_item["file"]
                dynamic_value = data_loader.load_as_matrix(file)
                facility_dynamic_data[item] = dynamic_value

            self.total_data.append({
                "facility_name" : facility_name,
                "shared_data"   : facility_shared_data, 
                "static_data"   : facility_static_data, 
                "dynamic_data"  : facility_dynamic_data
            })
    
    """
        Init and transform the shared, static and dynamic data by:
        - Remove useless sku in all sku data
        - Remove useless date in all sku data
        - Rename the date to step 
        - Check sku and date in static and dynamic data
        init_data will be called in reset function.
    """
    def init_data(self) -> None:
        self.picked_data = []
        for data in self.total_data:
            picked_facility_data = {"facility_name": data["facility_name"]}

            # Load shared info
            picked_facility_data["shared_data"] = data["shared_data"]

            # Load static info
            picked_facility_data["static_data"] = data["static_data"]
            assert(set(self.sku_list) <= set(list(data["static_data"]["SKU"].unique())))
            # Remove useless sku
            picked_facility_data["static_data"] = data["static_data"][data["static_data"]["SKU"].isin(self.sku_list)]

            # Load and check demands info, which is different between different skus and will change by date
            # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
            picked_facility_data["dynamic_data"] = {}
            for item, ori_dynamic_value in data["dynamic_data"].items():
                dynamic_value = copy.deepcopy(ori_dynamic_value)
                assert(set(self.sku_list) <= set(dynamic_value.columns.unique()))
                dates_format_dic = {
                    datetime.strftime(pd.to_datetime(date), "%Y/%m/%d"): date
                    for date in dynamic_value.index
                }
                self.date_to_index = {}
                # For warmup, start date forwards lookback_len
                for step, date in enumerate(pd.date_range(self.picked_start_date - timedelta(self.lookback_len), self.picked_end_date)):
                    date_str = datetime.strftime(date, "%Y/%m/%d")
                    # Check the target date in saved in demands file
                    assert(date_str in dates_format_dic.keys())
                    # Convert the actual date to date index
                    self.date_to_index[dates_format_dic[date_str]] = step - self.lookback_len
                dynamic_value.rename(index=self.date_to_index, inplace=True)
                # Remove useless date and sku
                dynamic_value = dynamic_value.loc[self.date_to_index.values()]
                dynamic_value = dynamic_value[self.sku_list]
                picked_facility_data["dynamic_data"][item] = dynamic_value.sort_values(by="Date")
            self.picked_data.append(picked_facility_data)

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update
    """
    def reset(self) -> None:
        self.init_env()
        self.init_data()
        self.init_state()
        self.init_monitor()
        eval(format(self.warmup_function))(self)
        states = self.get_state()
        return states  
    
    def init_env(self) -> None:
        # Get basic env info from config
        self.sku_count              = len(self.sku_list)
        self.facility_count         = self.supply_chain.get_facility_count()
        self.agent_count            = self.sku_count * self.facility_count
        self.integerization_sku     = self.config["env"].get("integerization_sku", False)
        self.lookback_len           = self.config["env"].get("lookback_len", 7)
        self.current_output_state   = self.config["output_state"].get("current_state", [])
        self.lookback_output_state  = self.config["output_state"].get("lookback_state", [])
        self.action_mode            = self.config["action"].get("mode", "continuous")
        self.warmup_function        = self.config["env"].get("warmup", "replenish_by_last_demand")
        self.balance                = [self.supply_chain[facility, "init_balance"] for facility in self.facility_list]
        self.per_balance            = np.zeros(self.agent_count)

        # Get mode related info from mode config
        mode_configs = [mode_config for mode_config in self.config["env"]["mode"] if mode_config["name"] == self.mode]
        assert(len(mode_configs) == 1)
        self.mode_config = mode_configs[0]
        assert("start_date" in self.mode_config)
        assert("end_date" in self.mode_config)
        self.start_date             = pd.to_datetime(self.mode_config["start_date"])
        self.end_date               = pd.to_datetime(self.mode_config["end_date"])
        self.random_interception    = self.mode_config.get("random_interception", False)

        # Step tick. Due to warmup, step starts from -self.lookback_len.
        self.current_step = -self.lookback_len

        self.action_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(self.agent_count, ), dtype=np.float32
        )
        # Current all agent_states will be returned as obs.
        output_length = len(self.current_output_state) + len(self.lookback_output_state) * self.lookback_len
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(output_length, len(self.sku_list)),
            dtype=np.float32
        )

        if self.random_interception:
            self.picked_start_date, self.picked_end_date = self.interception()
        else:
            self.picked_start_date, self.picked_end_date = self.start_date, self.end_date
        self.durations = (self.picked_end_date - self.picked_start_date).days + 1

    def interception(self) -> Tuple[datetime, datetime]:
        horizon = self.config["env"].get("horizon", 100)
        date_length = (self.end_date - self.start_date).days + 1
        start_date_index = random.randint(0, date_length - horizon)
        picked_start_date = self.start_date + timedelta(start_date_index)
        picked_end_date = picked_start_date + timedelta(horizon - 1)
        return picked_start_date, picked_end_date

    def init_state(self) -> None:
        self.agent_states = AgentStates(
            self.sku_list, 
            self.durations,
            self.picked_data,
            self.lookback_len,
        )

    def init_monitor(self) -> None:
        state_items = self.config["visualization"].get("state", ["replenish", "demand"])
        self.reward_info_list = []
        self.visualizer = Visualizer(
            self.agent_states, 
            self.reward_info_list,
            self.picked_start_date, 
            self.sku_list, 
            self.facility_list,
            state_items,
            self.vis_path,
            self.lookback_len
        )
        


    """
        Step orders: Replenish -> Sell -> Receive arrived skus -> Update balance
        actions: C * N matrix, C facility count, N agent count
        contains action_idx or action_quantity, defined by action_setting in config
    """
    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        self.replenish(actions)
        self.sell()
        self.receive_sku()
        self.profit, _ = self.get_reward()
        self.balance += np.sum(self.profit, axis=1)
        self.per_balance += self.profit.flatten()

        states = self.get_state()
        rewards, reward_info = self.get_reward()
        self.reward_info_list.append(reward_info)
        infos = self.get_info()
        infos["reward_info"] = reward_info

        self.next_step()
        done = self.current_step >= self.durations

        return states, rewards, done, infos

    """
        Sell by demand and in stock
    """
    def sell(self) -> None:
        current_demand = self.agent_states["all_facilities", "demand"]
        in_stock = self.agent_states["all_facilities", "in_stock"]
        sale = np.where(in_stock >= current_demand, current_demand, in_stock)
        self.agent_states["all_facilities", "sale"] = sale
        self.agent_states["all_facilities", "in_stock"] -= sale

        # Sold sku will arrive downstream facility after vlt dates 
        for facility in self.supply_chain.get_facility_list():
            if facility != self.supply_chain.get_tail():
                downstream = self.supply_chain[facility, "downstream"]
                vlt = self.agent_states[downstream, "vlt"]
                max_future = min(int(max(vlt)) + 1, self.durations - self.agent_states.current_step)
                arrived_index = np.array(range(0, max_future, 1)).reshape(-1, 1)
                arrived_flag = np.where(arrived_index == vlt, 1, 0)
                arrived_matrix = arrived_flag * self.agent_states[facility, "sale"]
                future_dates = np.array(range(0, max_future, 1)) + self.agent_states.current_step
                self.agent_states[downstream, "arrived", future_dates] += arrived_matrix
                self.agent_states[downstream, "in_transit"] += sale[self.facility_to_id[facility]]

    """
        Receive the arrived sku.
        When exceed the storage capacity, sku will be accepted in same ratio
    """
    def receive_sku(self) -> None:
        for facility in self.supply_chain.get_facility_list():
            # Arrived from upstream
            arrived = self.agent_states[facility, "arrived"]
            self.agent_states[facility, "in_transit"] -= arrived
            capacity = self.supply_chain[facility, "capacity"]
            accept_amount = eval(self.supply_chain[facility, "accept_sku"])(arrived, capacity, self.agent_states, facility)
            if self.integerization_sku:
                accept_amount = np.floor(accept_amount)
            
            self.agent_states[facility, "excess"] = arrived - accept_amount
            self.agent_states[facility, "accepted"] = accept_amount
            self.agent_states[facility, "in_stock"] += accept_amount

    """
        Replenish skus by actions
        actions: [action_idx/action_quantity] by sku order, defined by action setting in config
    """
    def replenish(self, actions) -> None:
        replenish_amount = eval(self.action_mode)(actions, self.config["action"], self.agent_states)

        if self.integerization_sku:
            replenish_amount = np.floor(replenish_amount)
        
        # Facility's replenishment is the replenishment as upstream's demand.
        for facility in self.supply_chain.get_facility_list():
            facility_replenish_amount = replenish_amount[self.facility_to_id[facility]]
            self.agent_states[facility, "replenish"] = facility_replenish_amount
            upstream = self.supply_chain[facility, "upstream"]
            if upstream == self.supply_chain.get_super_vendor():
                # If upstream is super_vendor, all replenishment will arrive after vlt dates
                vlt = self.agent_states[facility, "vlt"]
                max_future = min(int(max(vlt)) + 1, self.durations - self.agent_states.current_step)
                arrived_index = np.array(range(0, max_future, 1)).reshape(-1, 1)
                arrived_flag = np.where(arrived_index == vlt, 1, 0)
                arrived_matrix = arrived_flag * self.agent_states[facility, "replenish"]
                future_dates = np.array(range(0, max_future, 1)) + self.agent_states.current_step
                self.agent_states[facility, "arrived", future_dates] += arrived_matrix
                self.agent_states[facility, "in_transit"] += facility_replenish_amount
                
            else:
                # If upstream is not super_vendor, all replenishment will be sent as demand to upstream
                if self.agent_states.current_step < self.durations - 1:
                    self.agent_states[upstream, "demand"] = facility_replenish_amount

    def get_reward(self) -> Tuple[np.array, dict]:
        reward_info = {
            "unit_storage_cost": [self.supply_chain[facility, "unit_storage_cost"] for facility in self.facility_list]
        }
        reward_info = eval(self.config["reward_function"])(self.agent_states, reward_info)
        rewards = reward_info["reward"]
        return rewards, reward_info

    # Output C * N * M matrix:  C is facility count, N is sku count and M is state count
    def get_state(self) -> dict:
        states = self.agent_states.snapshot(self.current_output_state, self.lookback_output_state)
        return states

    def get_info(self) -> dict:
        info = {
            "profit": self.profit,
            "balance": self.balance
        }
        return info
    
    def pre_step(self) -> None:
        self.current_step -= 1
        self.agent_states.pre_step()

    def next_step(self) -> None:
        self.current_step += 1
        self.agent_states.next_step()

    def render(self) -> None:
        self.visualizer.render()
