import copy
from datetime import datetime, timedelta
from gym import Env, spaces
import numpy as np
import pandas as pd
import random
import yaml


from ReplenishmentEnv.utility.data_loader import DataLoader
from ReplenishmentEnv.env.reward_function.rewards import reward1, reward2
from ReplenishmentEnv.env.agent_states import AgentStates

class ReplenishmentEnv(Env):
    def __init__(self, config_path):
        # Config loaded from yaml file
        self.config: dict = None

        # Sku list from config file
        self.sku_list: list = None

        # Agents list. Each agent represents an sku.
        self.agents: list = None

        # Duration in days
        self.durations = 0

        # Look back information length
        self.lookback_len = 0

        # Agent state: object for AgentStates to save all agents info in current env step.
        # Inited from sku info file. Updated in env action.
        self.agent_states: AgentStates = None

        # 3 types of sku information which will be used in AgentStates init.
        # Shared info saved shared information for all skus and all dates.
        # Stored as dict = {state item: value} 
        self.shared_info: dict = None
        # Static info saved special information for each sku but will not changed.
        # Stored as N * M pd.DataFrame (N: agents_count, M: state item count)
        self.static_info: pd.DataFrame = None
        # Dynamic info saved the information which is different between different skus and will change by date.
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.dynamic_info: dict = None

        # Env balance
        self.balance = 0

        # Step tick.
        self.current_step = 0
 
        self.load_config(config_path)
        self.load_data()
        self.init_env()

    def load_config(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        assert("env" in self.config)
        assert("start_date" in self.config["env"])
        assert("end_date" in self.config["env"])
        assert("sku" in self.config)
        assert("sku_list" in self.config["sku"])
        assert("dynamic_info" in self.config["sku"])
        assert("static_info" in self.config["sku"])
        assert("shared_info" in self.config["sku"])
        assert("profit" in self.config)
        assert("reward" in self.config)
        assert("output_state" in self.config)
        assert("action" in self.config)

    """
        Load shared, static and dynamic data.
        Only load only in __init__ function.
    """
    def load_data(self) -> None:
        data_loader = DataLoader()
        # Convert the sku list from file to list. 
        if isinstance(self.config["sku"]["sku_list"], str):
            self.config["sku"]["sku_list"] = data_loader.load_as_list(self.config["sku"]["sku_list"])

        # Load shared info, which is shared for all skus and all dates.
        # Shared info is stored as dict = {state item: value}
        self.total_shared_info = self.config["sku"].get("shared_info", {})

        # Load and check static sku info, which is special for each sku but will not changed.
        # Static info is stored as N * M pd.DataFrame (N: agents_count, M: state item count)
        if "static_info" in self.config["sku"]:
            self.total_static_info = data_loader.load_as_df(self.config["sku"]["static_info"])
        else:
            self.total_static_info = np.zeros((len(self.agent_count), 0))

        # Load and check demands info, which is different between different skus and will change by date
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.total_dynamic_info = {}
        for dynamic_info_item in self.config["sku"].get("dynamic_info", {}):
            assert("name" in dynamic_info_item)
            assert("file" in dynamic_info_item)
            item = dynamic_info_item["name"]
            file = dynamic_info_item["file"]
            dynamic_value = data_loader.load_as_matrix(file)
            self.total_dynamic_info[item] = dynamic_value
    
    """
        Init and transform the shared, static and dynamic data by:
        - Remove useless sku in all sku data
        - Remove useless date in all sku data
        - Rename the date to step 
        - Check sku and date in static and dynamic data
        init_data will be called in reset function.
    """
    def init_data(self) -> None:
        # Load shared info
        self.shared_info = self.total_shared_info

        # Load static info
        self.static_info = self.total_static_info
        assert(set(self.sku_list) <= set(list(self.static_info["SKU"].unique())))
        # Remove useless sku
        self.static_info = self.static_info[self.static_info["SKU"].isin(self.sku_list)]

        # Load and check demands info, which is different between different skus and will change by date
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.dynamic_info = {}
        for item, ori_dynamic_value in self.total_dynamic_info.items():
            dynamic_value = copy.deepcopy(ori_dynamic_value)
            assert(set(self.sku_list) <= set(dynamic_value.columns.unique()))
            dates_format_dic = {
                datetime.strftime(pd.to_datetime(date), "%Y/%m/%d"): date
                for date in dynamic_value.index
            }
            self.durations = (self.picked_end_date - self.picked_start_date).days + 1
            self.date_to_index = {}
            for date_index, date in enumerate(pd.date_range(self.picked_start_date, self.picked_end_date)):
                date_str = datetime.strftime(date, "%Y/%m/%d")
                # Check the target date in saved in demands file
                assert(date_str in dates_format_dic.keys())
                # Convert the actual date to date index
                self.date_to_index[dates_format_dic[date_str]] = date_index
            dynamic_value.rename(index=self.date_to_index, inplace=True)
            # Remove useless date and sku
            dynamic_value = dynamic_value.loc[self.date_to_index.values()]
            dynamic_value = dynamic_value[self.sku_list]
            self.dynamic_info[item] = dynamic_value

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To update the sku_list: update_config = {"sku": {"sku_list": ["SKU0", "SKU1"]}}
        To update the start/end_date: update_config = {"env": {"start_date/end_date": 2022/01/01}} 
        To avoid the obfuscation, update_config is needed when reset with update
    """
    def reset(self, random_interception:bool = True) -> None:
        self.init_env()
        if random_interception:
            self.picked_start_date, self.picked_end_date = self.interception()
        else:
            self.picked_start_date, self.picked_end_date = self.start_date, self.end_date
             

        self.init_data()
        self.init_state()
        self.init_monitor()

        states = self.get_state()
        return states  
    
    def init_env(self) -> None:
        self.sku_list = self.config["sku"]["sku_list"]
        self.balance = self.config["env"].get("initial_balance", 0)
        self.storage_capacity = self.config["env"].get("storage_capacity", 0)
        self.integerization_sku = self.config["env"].get("integerization_sku", False)
        self.lookback_len = self.config["env"].get("lookback_len", 7)
        self.current_output_state = self.config["output_state"].get("current_state", [])
        self.lookback_output_state = self.config["output_state"].get("lookback_state", [])
        self.action_mode = self.config["action"].get("mode", "continuous")
        self.start_date = pd.to_datetime(self.config["env"]["start_date"])
        self.end_date = pd.to_datetime(self.config["env"]["end_date"])
        self.current_step = 0

        self.action_space = spaces.Box(
            low=0, high=np.inf, 
            shape=(len(self.sku_list), ), dtype=np.float32
        )
        # Current all agent_states will be returned as obs.
        output_length = len(self.current_output_state) + len(self.lookback_output_state) * self.lookback_len
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(output_length, len(self.sku_list)),
            dtype=np.float32
        )

    def interception(self) -> tuple[datetime, datetime]:
        horizon = self.config["env"].get("horizon", 100)
        date_length = (self.end_date - self.start_date).days + 1
        start_date_index = random.randint(0, date_length - horizon)
        picked_start_date = self.start_date + timedelta(start_date_index)
        picked_end_date = picked_start_date + timedelta(horizon - 1)
        return picked_start_date, picked_end_date

    def init_state(self) -> None:
        self.agents = self.sku_list
        self.agent_count = len(self.agents)
        self.agent_states = AgentStates(
            self.sku_list, 
            self.durations,
            self.dynamic_info,
            self.static_info, 
            self.shared_info,
            self.lookback_len,
        )

    def init_monitor(self) -> None:
        pass

    """
        Step orders: Replenish -> Sell -> Receive arrived skus -> Update balance
        actions: [action_idx/action_quantity] by sku order, defined by action_setting in config

    """
    def step(self, actions: np.array) -> tuple[np.array, np.array, list, dict]:

        self.replenish(actions)
        self.sell()
        self.receive_sku()
        self.profit = self.get_profit()
        self.balance += sum(self.profit)

        states = self.get_state()
        if self.config["reward"].get("mode", None) == "same_as_profit":
            rewards = self.profit
        else:
            rewards = self.get_reward()
        infos = self.get_info()

        self.next_step()
        done = self.current_step >= self.durations

        return states, rewards, done, infos

    """
        Sell by demand and in stock
    """
    def sell(self) -> None:
        current_demand = self.agent_states["demand"]
        in_stock = self.agent_states["in_stock"]
        self.agent_states["sale"] = np.where(in_stock >= current_demand, current_demand, in_stock)
        self.agent_states["in_stock"] -= self.agent_states["sale"]

    """
        Receive the arrived sku.
        When exceed the storage capacity, sku will be accepted in same ratio
    """
    def receive_sku(self) -> None:
        # calculate all arrived amount
        date_index = np.array(range(0, self.current_step, 1)).reshape(-1, 1)
        arrived_flag = np.where((self.agent_states["vlt", "history"] + date_index) == self.current_step, 1, 0)
        # Arrived for each sku
        arrived = np.sum(arrived_flag * self.agent_states["replenish", "history"], axis=0)  
        self.agent_states["in_transit"] -= arrived
        # Arrived for all skus
        total_arrived = sum(arrived)
        # Calculate accept ratio due to the capacity limitation.
        remaining_space = self.storage_capacity - np.sum(self.agent_states["in_stock"] * self.agent_states["volume"])
        accept_ratio = min(remaining_space / total_arrived, 1.0) if total_arrived > 0 else 0
        accept_amount = arrived * accept_ratio
        if self.integerization_sku:
            accept_amount = np.floor(accept_amount)
        # Receive skus by accept ratio
        self.agent_states["excess"] = arrived - accept_amount
        self.agent_states["in_stock"] += accept_amount

    """
        Replenish skus by actions
        actions: [action_idx/action_quantity] by sku order, defined by action setting in config
    """
    def replenish(self, actions) -> None:
        action_mode = self.config["action"]["mode"]
        if self.action_mode == "continuous":
            replenish_amount = actions
        elif self.action_mode == "discrete":
            assert("space" in self.config["action"])
            action_space = np.array(self.config["action"]["space"])
            replenish_amount = np.round(action_space[actions])
        elif self.action_mode == "demand_mean_continuous":
            # TODO: work around. Need to process look back.
            if self.current_step == 0:
                history_demand_mean = self.agent_states["demand"]
            else:
                history_demand_mean = np.average(self.agent_states["demand", "lookback"], 0)
            replenish_amount = actions * history_demand_mean
        elif self.action_mode == "demand_mean_discrete":
            # TODO: work around. Need to process look back.
            if self.current_step == 0:
                history_demand_mean = self.agent_states["demand"]
            else:
                history_demand_mean = np.average(self.agent_states["demand", "lookback"], 0)
            assert("space" in self.config["action"])
            action_space = np.array(self.config["action"]["space"])
            replenish_amount = action_space[actions] * history_demand_mean
        else:
            raise BaseException("No action mode {} found.".format(action_mode))

        if self.integerization_sku:
            replenish_amount = np.floor(replenish_amount)
        self.agent_states["replenish"] = replenish_amount
        self.agent_states["in_transit"] += replenish_amount

    def get_profit(self) -> np.array:
        profit_info = self.config["profit"]
        profit = eval(profit_info["profit_function"])(self.agent_states, profit_info)
        return profit

    def get_reward(self) -> np.array:
        reward_info = self.config["reward"]
        rewards = eval(reward_info["reward_function"])(self.agent_states, reward_info)
        return rewards

    # Ouput M * N matrix: M is state count and N is agent count
    def get_state(self) -> dict:
        states = self.agent_states.snapshot(self.current_output_state, self.lookback_output_state)
        return states

    def get_info(self) -> dict:
        info = {
            "profit": self.profit,
            "balance": self.balance
        }
        return info

    
    def next_step(self) -> None:
        self.current_step += 1
        self.agent_states.next_step()

    # TODO
    def render(self, mode: str="human", close: bool=False) -> None:
        pass
