from datetime import datetime
import copy
import numpy as np
import pandas as pd
import gym
import yaml


from ReplenishmentEnv.env.data_loader import DataLoader
from ReplenishmentEnv.env.reward_function.rewards import reward1, reward2
from ReplenishmentEnv.env.agent_states import AgentStates
from ReplenishmentEnv.utility.deep_update import deep_update

class ReplenishmentEnv(gym.Env):
    def __init__(self, config_path):
        # Config loaded from yaml file
        self.config: dict = None

        # Sku list from config file
        self.sku_list: list = None

        # Agents list. Each agent represents an sku.
        self.agents: list = None

        # Duration in days
        self.durations = 0

        # History information length
        self.history_len = 0

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

        # Step tick
        self.current_step = 0
 
        self.load_config(config_path)
        self.load_data()

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
        assert("store" in self.config)
        assert("balance" in self.config)
        assert("reward" in self.config)
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
        self.total_shared_info = self.config["sku"]["shared_info"]
        # Load and check static sku info, which is special for each sku but will not changed.
        # Static info is stored as N * M pd.DataFrame (N: agents_count, M: state item count)
        self.total_static_info = data_loader.load_as_df(self.config["sku"]["static_info"])
        # Load and check demands info, which is different between different skus and will change by date
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.total_dynamic_info = {}
        for dynamic_info_item in self.config["sku"]["dynamic_info"]:
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
        # Load sku list
        self.sku_list = self.config["sku"]["sku_list"]

        # Load shared info
        self.shared_info = self.total_shared_info

        # Load static info
        self.static_info = self.total_static_info
        assert(self.sku_list <= list(self.static_info["SKU"].unique()))
        # Remove useless sku
        self.static_info = self.static_info[self.static_info["SKU"].isin(self.sku_list)]

        # Load and check demands info, which is different between different skus and will change by date
        # Demands info stored as dict = {state_item: N * D pd.DataFrame (N: agents_count, D: dates)}
        self.dynamic_info = {}
        for item, ori_dynamic_value in self.total_dynamic_info.items():
            dynamic_value = copy.deepcopy(ori_dynamic_value)
            assert(self.sku_list <= list(dynamic_value.columns.unique()))
            dates_format_dic = {
                datetime.strftime(pd.to_datetime(date), "%Y/%m/%d"): date
                for date in dynamic_value.index
            }
            self.start_date = pd.to_datetime(self.config["env"]["start_date"])
            self.end_date = pd.to_datetime(self.config["env"]["end_date"])
            self.durations = (self.end_date - self.start_date).days + 1
            self.date_to_index = {}
            for date_index, date in enumerate(pd.date_range(self.start_date, self.end_date)):
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
    def reset(self, update_config:dict = None) -> None:
        if update_config is not None:
            self.config = deep_update(self.config, update_config)
        self.init_env()
        self.init_data()
        self.init_state()
        self.init_monitor()

        states = self.get_env_state()
        return states
        
    
    def init_env(self) -> None:
        self.balance = self.config["balance"].get("initial_balance", 0)
        self.storage_capacity = self.config["store"].get("storage_capacity", 0)
        self.unit_storage_cost = self.config["store"].get("unit_storage_cost", 1)
        self.backlog_ratio = self.config["env"].get("backlog_ratio", 0.01)
        self.action_mode = self.config["action"].get("mode", "discrete")
        self.action_precision = self.config["action"].get("precision", 2)
        self.action_space = np.array(self.config["action"].get("space", [0, 1]))
        self.history_len = self.config["env"].get("history_len", 1)
        self.current_step = 0
        self.integerization_sku = self.config["env"].get("integerization_sku", False)
    
    def init_state(self) -> None:
        self.agents = self.sku_list
        self.agent_count = len(self.agents)
        self.agent_states = AgentStates(
            self.sku_list, 
            self.durations,
            self.dynamic_info,
            self.static_info, 
            self.shared_info
        )

    def init_monitor(self) -> None:
        pass

    """
        Step orders: Replenish -> Sell -> Receive arrived skus -> Update balance
        actions: [action_idx/action_quantity] by sku order, defined by action_setting in config
    """
    def step(self, actions: np.array) -> None:
        self.replenish(actions)
        self.sell()
        self.receive_sku()
        self.current_balance = self.get_current_balance()
        self.balance += sum(self.current_balance)

        states = self.get_env_state()
        rewards = self.get_reward()
        dones = [self.current_step >= self.durations] * self.agent_count
        infos = self.get_info()

        self.next_step()
        return states, rewards, dones, infos

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
        actions: [action_idx/action_quantity] by sku order, defined by action_setting in config
    """
    def replenish(self, actions) -> None:
        history_demand_mean = np.average(self.agent_states["demand", "history_with_current"][-self.history_len:], 0)
        if self.action_mode == "discrete":
            replenish = self.action_space[actions] * history_demand_mean
        elif self.action_mode == "continuous":
            replenish = actions
        else:
            raise Exception("No action mode {} found. Only discrete and continuous are supported.".format(self.action_mode))
        if self.integerization_sku:
            replenish = np.round(replenish)
        self.agent_states["replenish"] = replenish

    def get_current_balance(self) -> np.array:
        env_info = {
            "current_demand": self.agent_states["demand"],
            "unit_storage_cost": self.unit_storage_cost,
            "backlog_ratio": self.backlog_ratio,
        }
        rewards = eval(self.config["balance"]["balance_function"])(self.agent_states, env_info)
        return rewards

    def get_reward(self) -> np.array:
        env_info = {
            "current_demand": self.agent_states["demand"],
            "unit_storage_cost": self.unit_storage_cost,
            "backlog_ratio": self.backlog_ratio,
        }
        rewards = eval(self.config["reward"]["reward_function"])(self.agent_states, env_info)
        return rewards
    
    # TODO
    def get_env_state(self) -> dict:
        return self.agent_states
        pass
    
    def get_info(self) -> dict:
        infos = {
            "current_balance": self.current_balance,
            "balance": self.balance
        }
        return infos

    
    def next_step(self) -> None:
        self.current_step += 1
        self.agent_states.next_step()

    # TODO
    def render(self, mode: str='human', close: bool=False) -> None:
        pass
