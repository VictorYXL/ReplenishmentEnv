import numpy as np

"""
Numpy based matrix to store the state for all stores and agents.
Use agent_states[facility_name, state_item, date, agent_id] to get the spaicel state for special agent and special day.
    - facility_name: Necessary item for target store. All facility_name can be found in get_all_stores().
    - state_item: Necessary item for target state. All state_item can be found in get_state_items().
    - date: set date to get state in special date.`
        - today: get state in current_step. Default is today.
        - yesterday: get state in current_step - 1.
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
            data: list=None, 
            lookback_len: int=7, 
        ) -> None:
        self.facility_list = [facility["facility_name"] for facility in data]
        self.agent_ids = agent_ids
        self.agents_count = len(self.agent_ids)
        self.states_items = self.get_state_items()
        self.inherited_states_items = self.get_inherited_items()
        self.init_0_items = self.get_init_0_items()

        # Look back len in date's look back mode.
        self.lookback_len = lookback_len

        # Step tick. Due to warmup, step starts from -lookback_len.
        self.current_step = -self.lookback_len

        # Durations length.
        self.durations = durations
    
        # Facility name to index dict.
        self.facility_to_index = {facility_name: index for index, facility_name in enumerate(self.facility_list)}

        # Agent id to index dict.
        self.agent_id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}

        # State to index dict.
        self.states_to_index = {state: index for index, state in enumerate(self.states_items)}
    
        # C * M * D * N: N , C facility count, M state item count, D dates count, N agent count
        self.states = np.zeros((
            len(self.facility_list),
            len(self.states_items),
            self.durations + self.lookback_len,
            len(self.agent_ids)
        ))

        # Init the state in order as: dynamic state, static_state, shared_state and default value
        for facility_data in data:
            for item in self.states_items:
                if item in facility_data["dynamic_data"]:
                    value = facility_data["dynamic_data"][item].to_numpy()
                elif item in facility_data["static_data"]:
                    value = facility_data["static_data"][item].to_numpy()
                elif item in facility_data["shared_data"]:
                    value = facility_data["shared_data"][item]
                elif item in self.init_0_items:
                    value = 0
                elif hasattr(self, "init_{0}".format(item)):
                    value = eval("self.init_{0}".format(item))(facility_data)
                else:
                    value = np.nan
                self.__setitem__([facility_data["facility_name"], item, "all_dates"], value)

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
            "unit_storage_cost",  # Unit storage cost per single sku
            "basic_holding_cost", # Fix basic holding cost for each sku
            "in_stock",           # Stock amount in current step
            "replenish",          # Replenish amount in current step for each sku
            "excess",             # Excess amount for each sku after sales
            "in_transit",         # Sku amount in transit 
            "arrived",            # Arrived amount from upstream
            "accepted",           # Accepted amount
            "backlog_ratio",      # Backlog = (selling_price - procurement_cost) * (cur_demand - sale) * backlog_ratio
        ]
        return states_items
    
    """
        The state that will inherite from yesterday
    """
    def get_inherited_items(self) -> list:
        states_items = [
            "in_stock",
            "in_transit",
        ]
        return states_items
    
    """
        The state that should be inited as 0
    """
    def get_init_0_items(self) -> list:
        states_items = [
            "demand",
            "in_transit",
            "arrived"
        ]
        return states_items

    # index = [facility_name, state, date, sku].
    # Facility_name and state are needed.
    def __getitem__(self, index):
        assert(isinstance(index, tuple) or isinstance(index, list))
        assert(len(index) >= 2 and len(index) <= 4)
        if len(index) == 2:
            facility_name = index[0]
            state_item = index[1]
            date = "today"
            agent_id = "all_agents"
        elif len(index) == 3:
            facility_name = index[0]
            state_item = index[1]
            date = index[2]
            agent_id = "all_agents"
        elif len(index) == 4:
            facility_name = index[0]
            state_item = index[1]
            date = index[2]
            agent_id = index[3]

        if facility_name == "all_facilities":
            facility_id = slice(None, None, None)
        else:
            assert(facility_name in self.facility_to_index)
            facility_id = self.facility_to_index[facility_name]

        if isinstance(date, str):
            # Due to warmup, today date index = current_step + lookback_len
            today = self.current_step + self.lookback_len
            if date == "today":
                date = today
            elif date == "yesterday":
                date = today - 1
            elif date == "history":
                date = slice(None, today, None)
            elif date == "history_with_current":
                date = slice(None, today + 1, None)
            elif date == "lookback":
                date = slice(max(today - self.lookback_len, 0), today, None)
            elif date == "lookback_with_current":
                date = slice(max(today - self.lookback_len + 1, 0), today + 1, None)
            elif date == "all_dates":
                date = slice(None, None, None)
        elif isinstance(date, int):
            date = date + self.lookback_len
        elif isinstance(date, np.ndarray):
            date = date + self.lookback_len


        if isinstance(date, int):
            assert(date >= 0 and date < self.durations + self.lookback_len)

        if agent_id == "all_agents":
            agent_index = slice(None, None, None)
        else:
            assert(agent_id in self.agent_id_to_index)
            agent_index = self.agent_id_to_index[agent_id]

        if state_item in self.states_items:
            state_index = self.states_to_index[state_item]
            return self.states[facility_id, state_index, date, agent_index]
        elif hasattr(self, "get_{0}".format(state_item)):
            return eval("self.get_{0}".format(state_item))(facility_id, date, agent_id)
        else:
            raise NotImplementedError

    # index = [facility_name, state, date, sku].
    # Facility_name and state are needed.
    def __setitem__(self, index, value):
        assert(isinstance(index, tuple) or isinstance(index, list))
        assert(len(index) >= 2 and len(index) <= 4)
        if len(index) == 2:
            facility_name = index[0]
            state_item = index[1]
            date = "today"
            agent_id = "all_agents"
        elif len(index) == 3:
            facility_name = index[0]
            state_item = index[1]
            date = index[2]
            agent_id = "all_agents"
        elif len(index) == 4:
            facility_name = index[0]
            state_item = index[1]
            date = index[2]
            agent_id = index[3]

        if facility_name == "all_facilities":
            facility_id = slice(None, None, None)
        else:
            assert(facility_name in self.facility_to_index)
            facility_id = self.facility_to_index[facility_name]
        
        if isinstance(date, str):
            # Due to warmup, today date index = current_step + lookback_len
            today = self.current_step + self.lookback_len
            if date == "today":
                date = today
            elif date == "yesterday":
                date = today - 1
            elif date == "history":
                date = slice(None, today, None)
            elif date == "history_with_current":
                date = slice(None, today + 1, None)
            elif date == "lookback":
                date = slice(max(today - self.lookback_len, 0), today, None)
            elif date == "lookback_with_current":
                date = slice(max(today - self.lookback_len + 1, 0), today + 1, None)
            elif date == "all_dates":
                date = slice(None, None, None)
        elif isinstance(date, int):
            date = date + self.lookback_len
        elif isinstance(date, np.ndarray):
            date = date + self.lookback_len

        if isinstance(date, int):
            assert(date >= 0 and date < self.durations + self.lookback_len)

        if agent_id == "all_agents":
            agent_index = slice(None, None, None)
        else:
            assert(agent_id in self.agent_id_to_index)
            agent_index = self.agent_id_to_index[agent_id]
            
        if state_item in self.states_items:
            state_index = self.states_to_index[state_item]
            self.states[facility_id, state_index, date, agent_index] = value
        elif hasattr(self, "set_{0}".format(state_item)):
            eval("self.set_{0}".format(state_item))(value, facility_id, date, agent_id)
        else:
            raise NotImplementedError

    def next_step(self):
        self.current_step += 1
        # Inherited states items
        if self.current_step < self.durations:
            for item in self.inherited_states_items:
                self.__setitem__(["all_facilities", item], self.__getitem__(["all_facilities", item, "yesterday"]))
    
    def pre_step(self):
        self.current_step -= 1
    
    """
        Init in_stock from init_stock
    """
    def init_in_stock(self, facility_data: dict) -> np.array:
        # First date
        first_value = facility_data["static_data"].get("init_stock", 0).to_numpy().reshape(1, -1)
        # Rest dates
        rest_value = (np.ones((self.durations + self.lookback_len - 1, self.agents_count)) * np.nan)
        value = np.concatenate([first_value, rest_value])
        return value
    
    # Output M * N matrix: M is state count and N is agent count
    # TODO: Update to multi-facility_list
    def snapshot(self, current_state_items, lookback_state_items):
        states_list = [self.__getitem__(item).reshape(1, self.agents_count).copy() for item in current_state_items]
        for item in lookback_state_items:
            state = self.__getitem__([item, "lookback_with_current"]).copy()
            states_list.append(state)
        states = np.concatenate(states_list)
        return states
        
