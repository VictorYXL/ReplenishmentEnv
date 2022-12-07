"""
    SupplyChain class, contains a list of facilities.
    Currently, only chainlike supply chain are supprted. 
    Each facility has only 1 upstream and 1 downstream.
    Head facility receive skus from super_vendor and tail facility sells skus to consumer.
"""
class SupplyChain:
    def __init__(self, facility_config) -> None:
        """
            self.facility_dict = {
                facility_name: [upstream_name, downstream_name, capacity],
                facility_name: [upstream_name, downstream_name, capacity],
                ...
            }
            Facility's index represents the order as actions, rewards, and agent_states in env.
        """
        self.facility_dict = {}
        self.head = None
        self.tail = None
        self.consumer = "consumer"
        self.super_vendor = "super_vendor"
        for index, facility in enumerate(facility_config):
            assert("name" in facility)
            assert("upstream" in facility)
            assert("downstream" in facility)
            assert("sku" in facility)
            if facility["upstream"] == self.super_vendor:
                self.head = facility["name"]
            if facility["downstream"] == self.consumer:
                self.tail = facility["name"]
            self.facility_dict[facility["name"]] = {
                "upstream": facility["upstream"],
                "downstream": facility["downstream"],
            }

        # Check for supply chain
        assert(self.head is not None)
        assert(self.tail is not None)
        for facility, value in self.facility_dict.items():
            assert(isinstance(value["upstream"], str))
            assert(isinstance(value["upstream"], str))
            if self.facility_dict[facility]["upstream"] != "super_vendor":
                assert(self.facility_dict[self.facility_dict[facility]["upstream"]]["downstream"] == facility)
            if self.facility_dict[facility]["downstream"] != "consumer":
                assert(self.facility_dict[self.facility_dict[facility]["downstream"]]["upstream"] == facility)

    # index = [facility_name, item].
    # item includes upstream, downstream and capacity
    def __getitem__(self, index):
        assert(isinstance(index, tuple))
        assert(len(index) == 2)
        facility_name = index[0]
        item = index[1]
        assert(facility_name in self.facility_dict)
        assert(item in self.facility_dict[facility_name])
        return self.facility_dict[facility_name][item]
    
    # Return the head index, which upstream is super_vendor
    def get_head(self) -> int:
        return self.head

    # Return the tail index, which downstream is consumer
    def get_tail(self) -> int:
        return self.tail

    def get_facility_count(self) -> int:
        return len(self.facility_dict)

    def get_facility_list(self) -> list:
        return list(self.facility_dict.keys())
    
    def get_consumer(self) -> str:
        return self.consumer
    
    def get_super_vendor(self) -> str:
        return self.super_vendor