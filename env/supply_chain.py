"""
    SupplyChain class, contains a list of warehouses.
    Currently, only chainlike supply chain are supported. 
    Each warehouse has only 1 upstream and 1 downstream.
    Head warehouse receive skus from super_vendor and tail warehouse sells skus to consumer.
"""
class SupplyChain:
    def __init__(self, warehouse_config) -> None:
        """
            self.warehouse_dict = {
                warehouse_name: [upstream_name, downstream_name, capacity],
                warehouse_name: [upstream_name, downstream_name, capacity],
                ...
            }
            Warehouse's index represents the order as actions, rewards, and agent_states in env.
        """
        self.warehouse_dict = {}
        self.head = None
        self.tail = None
        self.consumer = "consumer"
        self.super_vendor = "super_vendor"
        for index, warehouse in enumerate(warehouse_config):
            assert("name" in warehouse)
            assert("upstream" in warehouse)
            assert("downstream" in warehouse)
            assert("sku" in warehouse)
            if warehouse["upstream"] == self.super_vendor:
                self.head = warehouse["name"]
            if warehouse["downstream"] == self.consumer:
                self.tail = warehouse["name"]
            self.warehouse_dict[warehouse["name"]] = {
                "upstream": warehouse["upstream"],
                "downstream": warehouse["downstream"],
                "capacity": warehouse.get("capacity", 0),
                "init_balance": warehouse.get("init_balance", 0),
                "unit_storage_cost": warehouse.get("unit_storage_cost", 0),
                "accept_sku": warehouse.get("accept_sku", "equal_accept")
            }

        # Check for supply chain
        assert(self.head is not None)
        assert(self.tail is not None)
        for warehouse, value in self.warehouse_dict.items():
            assert(isinstance(value["upstream"], str))
            assert(isinstance(value["upstream"], str))
            if self.warehouse_dict[warehouse]["upstream"] != "super_vendor":
                assert(self.warehouse_dict[self.warehouse_dict[warehouse]["upstream"]]["downstream"] == warehouse)
            if self.warehouse_dict[warehouse]["downstream"] != "consumer":
                assert(self.warehouse_dict[self.warehouse_dict[warehouse]["downstream"]]["upstream"] == warehouse)

    # index = [warehouse_name, item].
    # item includes upstream, downstream, capacity, init_balance and unit_storage_cost
    def __getitem__(self, index):
        assert(isinstance(index, tuple))
        assert(len(index) == 2)
        warehouse_name = index[0]
        item = index[1]
        assert(warehouse_name in self.warehouse_dict)
        assert(item in self.warehouse_dict[warehouse_name])
        return self.warehouse_dict[warehouse_name][item]
    
    # Return the head index, which upstream is super_vendor
    def get_head(self) -> int:
        return self.head

    # Return the tail index, which downstream is consumer
    def get_tail(self) -> int:
        return self.tail

    def get_warehouse_count(self) -> int:
        return len(self.warehouse_dict)

    def get_warehouse_list(self) -> list:
        return list(self.warehouse_dict.keys())
    
    def get_consumer(self) -> str:
        return self.consumer
    
    def get_super_vendor(self) -> str:
        return self.super_vendor
