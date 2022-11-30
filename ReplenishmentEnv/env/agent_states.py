import numpy as np

"""
Numpy based matrix to store the state for all agents.
Use agent_states[state_item, date, agent_id] to get the spaicel state for special agent and special day.
    - state_item: is necessary.
    - date: can be set to today/history/history_with_current/all_dates to
            get the information for [today]/[all history days]/[all history + current days]/[all days].
            Default is today.
    - agent_id: can be set to "all_agents" to get state for all skus. 
            Default is all_agents.
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
        ) -> None:
        self.agent_ids = agent_ids
        self.agents_count = len(self.agent_ids)
        self.states_items = self.get_state_items()
        self.inherited_states_items = self.get_inherited_items()
        self.agent_id_to_index = {agent_id: index for index, agent_id in enumerate(self.agent_ids)}
        self.states_to_index = {state: index for index, state in enumerate(self.states_items)}
        # M * D * N: N , M state item count, D dates count, N agent count
        self.states = np.zeros((len(self.states_items), durations, self.agents_count))
        self.current_step = 0
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
            "selling_price",    # Sku sale price to downstream
            "procurement_cost", # Sku buy in cost from upstream
            "demand",           # Sku demand amount for consumer
            "sale",             # SKU sale amount in env 
            "vlt",              # Fixed vendor leading time for each sku
            "volume",           # Sku volume, represents how much storage this sku costs.
            "order_cost",       # Cost for each order
            "in_stock",         # Stock amount in current step
            "replenish",        # Replenish amount in current step for each sku
            "excess",           # Excess amount for each sku after sales
            "in_transit",       # Sku amount in transit 
            "excess",           # Amount that sku exceeded the capacity for storage
        ]
        return states_items
    
    """
        The value that will inherite from yesterday
    """
    def get_inherited_items(self) -> list:
        states_items = [
            "in_stock",
        ]
        return states_items

    # index = [state, sku, date].
    # state is needed. 
    # sku is defualt to all skus.
    # date is defualt to current step.
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

        if date == "all_dates":
            date = slice(None, None, None)
        elif date == "history":
            date = slice(None, self.current_step, None)
        elif date == "history_with_current":
            date = slice(None, self.current_step + 1, None)
        elif date == "today":
            date = self.current_step

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
    # sku is defualt to all skus.
    # date is defualt to current step.
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

        if date == "all_dates":
            date = slice(None, None, None)
        elif date == "history":
            date = slice(None, self.current_step, None)
        elif date == "history_with_current":
            date = slice(None, self.current_step + 1, None)
        elif date == "today":
            date = self.current_step

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
        if self.current_step < self.durations:
            for item in self.inherited_states_items:
                self.__setitem__(item, self.__getitem__([item, "history"])[-1])
    
    """
        Init in_stock from init_stock
    """
    def init_in_stock(self,             
            dynamic_info: dict=None, 
            static_info: np.array=None, 
            shared_info: dict=None):
        # First date
        first_value = static_info.get("init_stock", 0).to_numpy().reshape(1, -1)
        # Rest dates
        rest_value = (np.ones((self.durations - 1, self.agents_count)) * np.nan)
        value = np.concatenate([first_value, rest_value])
        return value

