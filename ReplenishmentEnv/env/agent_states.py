import numpy as np

"""
Numpy based matrix to store the state for all agents.
Use agent_states[state_item, date, agent_id] to get the spaicel state for special agent and special day.
    - state_item: set state_item to get target state_item. All state_item can be found in get_state_items().
    - date: set date to get state in special date.
        - today: get state in current_step. Default is today.
        - history: get state in all history days. 
        - history_with_current: get state in all history days with current step.
        - lookback: get state in lookback_len history days.
        - lookback_with_current: get state in lookback_len - 1 history days with current step.
        - all_dates: get state for all dataes.
    - agent_id: set agent_id to get the target agent info.
        - all_agents: get state for all skus. Default is all_agents.
For the state which is not stated in state matrix, difinite the get_/set_/init_ function to realize it.
"""
class AgentStates(object):
    def __init__(
            self, 
            agent_ids: list, 
            durations: int=0,
            dynamic_info: dict=None, 
            static_info: np.array=None, 
            shared_info: dict=None, 
            lookback_len: int=7, 
        ) -> None:
        self.agent_ids = agent_ids
        self.agents_count = len(self.agent_ids)
        self.states_items = self.get_state_items()
        self.inherited_states_items = self.get_inherited_items()

        # Agent id to index dict.
        self.agent_id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}

        # State to index dict.
        self.states_to_index = {state: index for index, state in enumerate(self.states_items)}
    
        # M * D * N: N , M state item count, D dates count, N agent count
        self.states = np.zeros((len(self.states_items), durations, self.agents_count))
        
        # Step tick.
        self.current_step = 0

        # Look back len in date's look back mode.
        self.lookback_len = lookback_len

        # Durations length.
        self.durations = durations

        # Init the state in order as: dynamic state, static_state, shared_state and default value
        for item in self.states_items:
            if item in dynamic_info:
                value = dynamic_info[item].to_numpy()
            elif item in static_info:
                value = static_info[item].to_numpy()
            elif item in shared_info:
                value = shared_info[item]
            elif hasattr(self, "init_{0}".format(item)):
                value = eval("self.init_{0}".format(item))(dynamic_info, static_info, shared_info)
            else:
                value = np.nan
            self.__setitem__([item, "all_dates"], value)

    def __len__ (self):
        return len(self.agent_ids)

    """
        All state items values
    """
    def get_state_items(self) -> list:
        states_items = [
            "selling_price",      # Sku sale price to downstream
            "procurement_cost",   # Sku buy in cost from upstream
            "demand",             # Sku demand amount for consumer
            "sale",               # SKU sale amount in env 
            "vlt",                # Fixed vendor leading time for each sku
            "volume",             # Sku volume, represents how much storage this sku costs.
            "order_cost",         # Cost for each order
            "storage_cost",       # Storage cost per single sku
            "holding_cost_ratio", # holding_cost = selling_price * holding_cost_ratio + storage_cost
            "in_stock",           # Stock amount in current step
            "replenish",          # Replenish amount in current step for each sku
            "excess",             # Excess amount for each sku after sales
            "in_transit",         # Sku amount in transit 
            "excess",             # Amount that sku exceeded the capacity for storage
        ]
        return states_items
    
    """
        The value that will inherite from yesterday
    """
    def get_inherited_items(self) -> list:
        states_items = [
            "in_stock",
            "in_transit",
        ]
        return states_items

    # index = [state, sku, date].
    # state is needed.
    def __getitem__(self, index):
        if isinstance(index, str):
            state_item = index
            date = "today"
            agent_id = "all_agents"
        elif len(index) == 2:
            state_item = index[0]
            date = index[1]
            agent_id = "all_agents"
        elif len(index) == 3:
            state_item = index[0]
            date = index[1]
            agent_id = index[2]

        if date == "today":
            date = self.current_step
        elif date == "history":
            date = slice(None, self.current_step, None)
        elif date == "history_with_current":
            date = slice(None, self.current_step + 1, None)
        elif date == "lookback":
            date = slice(max(self.current_step - self.lookback_len, 0), self.current_step, None)
        elif date == "lookback_with_current":
            date = slice(max(self.current_step - self.lookback_len + 1, 0), self.current_step + 1, None)
        elif date == "all_dates":
            date = slice(None, None, None)


        if agent_id == "all_agents":
            agent_index = slice(None, None, None)
        else:
            agent_index = self.agent_id_to_index[agent_id]

        if state_item in self.states_items:
            state_index = self.states_to_index[state_item]
            return self.states[state_index, date, agent_index]
        elif hasattr(self, "get_{0}".format(state_item)):
            return eval("self.get_{0}".format(state_item)(date, agent_id))
        else:
            raise NotImplementedError

    # index = [state, sku, date].
    # state is needed.
    def __setitem__(self, index, value):
        if isinstance(index, str):
            state_item = index
            date = "today"
            agent_id = "all_agents"
        elif len(index) == 2:
            state_item = index[0]
            date = index[1]
            agent_id = "all_agents"
        elif len(index) == 3:
            state_item = index[0]
            date = index[1]
            agent_id = index[2]

        if date == "today":
            date = self.current_step
        elif date == "history":
            date = slice(None, self.current_step, None)
        elif date == "history_with_current":
            date = slice(None, self.current_step + 1, None)
        elif date == "lookback":
            date = slice(self.current_step - self.lookback_len, self.current_step, None)
        elif date == "lookback_with_current":
            date = slice(self.current_step - self.lookback_len, self.current_step + 1, None)
        elif date == "all_dates":
            date = slice(None, None, None)

        if agent_id == "all_agents":
            agent_index = slice(None, None, None)
        else:
            agent_index = self.agent_id_to_index[agent_id]
            
        if state_item in self.states_items:
            state_index = self.states_to_index[state_item]

            self.states[state_index, date, agent_index] = value
        elif hasattr(self, "set_{0}".format(state_item)):
            eval("self.set_{0}".format(state_item)(value, date, agent_id))
        else:
            raise NotImplementedError

    def next_step(self):
        self.current_step += 1
        # Inherited states items
        if self.current_step < self.durations:
            for item in self.inherited_states_items:
                self.__setitem__(item, self.__getitem__([item, "history"])[-1])
    
    """
        Init in_stock from init_stock
    """
    def init_in_stock(self,             
            dynamic_info: dict=None, 
            static_info: np.array=None, 
            shared_info: dict=None) -> np.array:
        # First date
        first_value = static_info.get("init_stock", 0).to_numpy().reshape(1, -1)
        # Rest dates
        rest_value = (np.ones((self.durations - 1, self.agents_count)) * np.nan)
        value = np.concatenate([first_value, rest_value])
        return value
    
    def init_in_transit(self,
            dynamic_info: dict=None, 
            static_info: np.array=None, 
            shared_info: dict=None) -> np.array:
        return 0
    
    # Ouput M * N matrix: M is state count and N is agent count
    def snapshot(self, current_state_items, lookback_state_items):
        states_list = [self.__getitem__(item).reshape(1, self.agents_count) for item in current_state_items]
        for item in lookback_state_items:
            state = np.zeros((self.lookback_len, self.agents_count))
            state[-self.current_step-1:] = self.__getitem__([item, "lookback_with_current"])
            states_list.append(state)
        states = np.concatenate(states_list)
        return states
        
