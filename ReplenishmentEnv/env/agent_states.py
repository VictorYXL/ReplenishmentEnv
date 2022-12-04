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
            dynamic_data: dict=None, 
            static_data: np.array=None, 
            shared_data: dict=None, 
            lookback_len: int=7, 
        ) -> None:
        self.agent_ids = agent_ids
        self.agents_count = len(self.agent_ids)
        self.states_items = self.get_state_items()
        self.inherited_states_items = self.get_inherited_items()

        # Look back len in date's look back mode.
        self.lookback_len = lookback_len

        # Step tick. Due to warmup, step starts from -lookback_len.
        self.current_step = -self.lookback_len

        # Durations length.
        self.durations = durations
    
        # Agent id to index dict.
        self.agent_id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}

        # State to index dict.
        self.states_to_index = {state: index for index, state in enumerate(self.states_items)}
    
        # M * D * N: N , M state item count, D dates count, N agent count
        self.states = np.zeros((len(self.states_items), self.durations + self.lookback_len, self.agents_count))

        # Init the state in order as: dynamic state, static_state, shared_state and default value
        for item in self.states_items:
            if item in dynamic_data:
                value = dynamic_data[item].to_numpy()
            elif item in static_data:
                value = static_data[item].to_numpy()
            elif item in shared_data:
                value = shared_data[item]
            elif hasattr(self, "init_{0}".format(item)):
                value = eval("self.init_{0}".format(item))(dynamic_data, static_data, shared_data)
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

    # index = [state, date, sku].
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

        # Due to warmup, date_index = current_step + lookback_len
        date_index = self.current_step + self.lookback_len
        if date == "today":
            date = date_index
        elif date == "history":
            date = slice(None, date_index, None)
        elif date == "history_with_current":
            date = slice(None, date_index + 1, None)
        elif date == "lookback":
            date = slice(max(date_index - self.lookback_len, 0), date_index, None)
        elif date == "lookback_with_current":
            date = slice(max(date_index - self.lookback_len + 1, 0), date_index + 1, None)
        elif date == "all_dates":
            date = slice(None, None, None)

        if isinstance(date, int):
            assert(date >= 0 and date < self.durations + self.lookback_len)

        if agent_id == "all_agents":
            agent_index = slice(None, None, None)
        else:
            assert(agent_id in self.agent_id_to_index)
            agent_index = self.agent_id_to_index[agent_id]

        if state_item in self.states_items:
            state_index = self.states_to_index[state_item]
            return self.states[state_index, date, agent_index]
        elif hasattr(self, "get_{0}".format(state_item)):
            return eval("self.get_{0}".format(state_item)(date, agent_id))
        else:
            raise NotImplementedError

    # index = [state, date, sku].
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

        # Due to warmup, date_index = current_step + lookback_len
        date_index = self.current_step + self.lookback_len
        if date == "today":
            date = date_index
        elif date == "history":
            date = slice(None, date_index, None)
        elif date == "history_with_current":
            date = slice(None, date_index + 1, None)
        elif date == "lookback":
            date = slice(max(date_index - self.lookback_len, 0), date_index, None)
        elif date == "lookback_with_current":
            date = slice(max(date_index - self.lookback_len + 1, 0), date_index + 1, None)
        elif date == "all_dates":
            date = slice(None, None, None)

        if isinstance(date, int):
            assert(date >= 0 and date < self.durations + self.lookback_len)

        if agent_id == "all_agents":
            agent_index = slice(None, None, None)
        else:
            assert(agent_id in self.agent_id_to_index)
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
    
    def pre_step(self):
        self.current_step -= 1
    
    """
        Init in_stock from init_stock
    """
    def init_in_stock(self,             
            dynamic_data: dict=None, 
            static_data: np.array=None, 
            shared_data: dict=None) -> np.array:
        # First date
        first_value = static_data.get("init_stock", 0).to_numpy().reshape(1, -1)
        # Rest dates
        rest_value = (np.ones((self.durations + self.lookback_len - 1, self.agents_count)) * np.nan)
        value = np.concatenate([first_value, rest_value])
        return value
    
    def init_in_transit(self,
            dynamic_data: dict=None, 
            static_data: np.array=None, 
            shared_data: dict=None) -> np.array:
        return 0
    
    # Output M * N matrix: M is state count and N is agent count
    def snapshot(self, current_state_items, lookback_state_items):
        states_list = [self.__getitem__(item).reshape(1, self.agents_count).copy() for item in current_state_items]
        for item in lookback_state_items:
            state = self.__getitem__([item, "lookback_with_current"]).copy()
            states_list.append(state)
        states = np.concatenate(states_list)
        return states
        
