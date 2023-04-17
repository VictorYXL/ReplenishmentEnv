import numpy as np

"""
Numpy based matrix to store the state for all stores and skus.
Use sku_states[warehouse_name, state_item, date, sku_id] to get the states for special sku and special day.
    - warehouse_name: Necessary item for target store. All warehouse_name can be found in get_all_stores().
    - state_item: Necessary item for target state. All state_item can be found in get_state_items().
    - date: set date to get state in special date.`
        - today: get state in current_step. Default is today.
        - yesterday: get state in current_step - 1.
        - history: get state in all history days. 
        - history_with_current: get state in all history days with current step.
        - lookback: get state in lookback_len history days.
        - lookback_with_current: get state in lookback_len - 1 history days with current step.
        - all_dates: get state for all dataes.
    - sku_id: set sku_id to get the target sku info.
        - all_skus: get state for all skus. Default is all_skus.
For the state which is not stated in state matrix, definite the get_/set_/init_ function to realize it.
"""
class AgentStates(object):
    def __init__(
            self, 
            sku_ids: list, 
            durations: int=0,
            data: list=None, 
            lookback_len: int=7, 
        ) -> None:
        self.warehouse_list = [warehouse["warehouse_name"] for warehouse in data]
        self.sku_ids = sku_ids
        self.skus_count = len(self.sku_ids)
        self.states_items = self.get_state_items()
        self.inherited_states_items = self.get_inherited_items()
        self.init_0_items = self.get_init_0_items()

        # Look back len in date's look back mode.
        self.lookback_len = lookback_len

        # Step tick. Due to warmup, step starts from -lookback_len.
        self.current_step = -self.lookback_len

        # Durations length.
        self.durations = durations
    
        # Warehouse name to index dict.
        self.warehouse_to_id = {warehouse_name: index for index, warehouse_name in enumerate(self.warehouse_list)}

        # Sku name to index dict.
        self.sku_to_id = {sku_id: index for index, sku_id in enumerate(self.sku_ids)}

        # State to index dict.
        self.states_to_id = {state: index for index, state in enumerate(self.states_items)}
    
        # C * M * D * N: N , C warehouse count, M state item count, D dates count, N sku count
        self.states = np.zeros((
            len(self.warehouse_list),
            len(self.states_items),
            self.durations + self.lookback_len,
            len(self.sku_ids)
        ))

        # Init the state in order as: dynamic state, static_state, shared_state and default value
        for warehouse_data in data:
            for item in self.states_items:
                if item in warehouse_data["dynamic_data"]:
                    value = warehouse_data["dynamic_data"][item].to_numpy()
                elif item in warehouse_data["static_data"]:
                    value = warehouse_data["static_data"][item].to_numpy()
                elif item in warehouse_data["shared_data"]:
                    value = warehouse_data["shared_data"][item]
                elif item in self.init_0_items:
                    value = 0
                elif hasattr(self, "init_{0}".format(item)):
                    value = eval("self.init_{0}".format(item))(warehouse_data)
                else:
                    value = np.nan
                self.__setitem__([warehouse_data["warehouse_name"], item, "all_dates"], value)

    def __len__ (self):
        return len(self.sku_ids)

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
            "unit_order_cost",    # Cost for single order
            "basic_holding_cost", # Fix basic holding cost for each sku
            "in_stock",           # Stock amount in current step
            "replenish",          # Replenish amount in current step for each sku
            "excess",             # Excess amount for each sku after sales
            "excess_ratio",       # Cost ratio due to excess
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

    # index = [warehouse_name, state, date, sku].
    # Warehouse_name and state are needed.
    def __getitem__(self, index):
        assert(isinstance(index, tuple) or isinstance(index, list))
        assert(len(index) >= 2 and len(index) <= 4)
        if len(index) == 2:
            warehouse_name = index[0]
            state_item = index[1]
            date = "today"
            sku_id = "all_skus"
        elif len(index) == 3:
            warehouse_name = index[0]
            state_item = index[1]
            date = index[2]
            sku_id = "all_skus"
        elif len(index) == 4:
            warehouse_name = index[0]
            state_item = index[1]
            date = index[2]
            sku_id = index[3]

        if warehouse_name == "all_warehouses":
            warehouse_id = slice(None, None, None)
        else:
            assert(warehouse_name in self.warehouse_to_id)
            warehouse_id = self.warehouse_to_id[warehouse_name]

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

        if sku_id == "all_skus":
            sku_index = slice(None, None, None)
        else:
            assert(sku_id in self.sku_to_id)
            sku_index = self.sku_to_id[sku_id]

        if state_item in self.states_items:
            state_index = self.states_to_id[state_item]
            return self.states[warehouse_id, state_index, date, sku_index]
        elif hasattr(self, "get_{0}".format(state_item)):
            return eval("self.get_{0}".format(state_item))(warehouse_id, date, sku_id)
        else:
            raise NotImplementedError

    # index = [warehouse_name, state, date, sku].
    # Warehouse_name and state are needed.
    def __setitem__(self, index, value):
        assert(isinstance(index, tuple) or isinstance(index, list))
        assert(len(index) >= 2 and len(index) <= 4)
        if len(index) == 2:
            warehouse_name = index[0]
            state_item = index[1]
            date = "today"
            sku_id = "all_skus"
        elif len(index) == 3:
            warehouse_name = index[0]
            state_item = index[1]
            date = index[2]
            sku_id = "all_skus"
        elif len(index) == 4:
            warehouse_name = index[0]
            state_item = index[1]
            date = index[2]
            sku_id = index[3]

        if warehouse_name == "all_warehouses":
            warehouse_id = slice(None, None, None)
        else:
            assert(warehouse_name in self.warehouse_to_id)
            warehouse_id = self.warehouse_to_id[warehouse_name]
        
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

        if sku_id == "all_skus":
            sku_index = slice(None, None, None)
        else:
            assert(sku_id in self.sku_to_id)
            sku_index = self.sku_to_id[sku_id]
            
        if state_item in self.states_items:
            state_index = self.states_to_id[state_item]
            self.states[warehouse_id, state_index, date, sku_index] = value
        elif hasattr(self, "set_{0}".format(state_item)):
            eval("self.set_{0}".format(state_item))(value, warehouse_id, date, sku_id)
        else:
            raise NotImplementedError

    def next_step(self):
        self.current_step += 1
        # Inherited states items
        if self.current_step < self.durations:
            for item in self.inherited_states_items:
                self.__setitem__(["all_warehouses", item], self.__getitem__(["all_warehouses", item, "yesterday"]))
    
    def pre_step(self):
        self.current_step -= 1
    
    """
        Init in_stock from init_stock
    """
    def init_in_stock(self, warehouse_data: dict) -> np.array:
        # First date
        first_value = warehouse_data["static_data"].get("init_stock", 0).to_numpy().reshape(1, -1)
        # Rest dates
        rest_value = (np.ones((self.durations + self.lookback_len - 1, self.skus_count)) * np.nan)
        value = np.concatenate([first_value, rest_value])
        return value
    
    # # Output C * N * M matrix:  C is warehouse count, N is sku count and M is state count
    def snapshot(self, current_state_items, lookback_state_items):
        states_list = []
        for warehouse in self.warehouse_list:
            single_warehouse_states_list = [self.__getitem__([warehouse, item]).reshape(1, self.skus_count, 1).copy() for item in current_state_items]
            for item in lookback_state_items:
                state = self.__getitem__([warehouse, item, "lookback_with_current"]).copy()
                single_warehouse_states_list.append(state.reshape(1, self.skus_count, -1))
            single_warehouse_states = np.concatenate(single_warehouse_states_list, axis=-1)
            states_list.append(single_warehouse_states)
        states = np.concatenate(states_list, axis=0)
        return states
        
