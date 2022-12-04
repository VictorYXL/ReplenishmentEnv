"""
    SupplyChain class, contains a lift of facilities.
    Currently, only chainlike supply chain are supprted. 
    Each facility has only 1 upstream and 1 downstream.
    Head facility receive skus from super_vendor and tail facility sells skus to consumer.
"""
class SupplyChain:
    def __init__(self, facility_config) -> None:
        """
            self.facilities = {
                facility_name: [upstream_name, downstream_name],
                facility_name: [upstream_name, downstream_name],
                ...
            }
            Facility's index represents the order as actions, rewards, and agent_states in env.
        """
        self.facilities = {}
        self.head = None
        self.tail = None
        for index, facility in enumerate(facility_config):
            assert("name" in facility)
            assert("upstream" in facility)
            assert("downstream" in facility)
            assert("sku" in facility)
            if facility["upstream"] == "super_vendor":
                self.head = facility["name"]
            if facility["downstream"] == "consumer":
                self.tail = facility["name"]
            self.facilities[facility["name"]] = [facility["upstream"], facility["downstream"]]

        assert(self.head is not None)
        assert(self.tail is not None)
        for facility in self.facilities:
            assert(isinstance(facility[0], str))
            assert(isinstance(facility[1], str))
            if self.facilities[facility][0] != "super_vendor":
                assert(self.facilities[self.facilities[facility][0]][1] == facility)
            if self.facilities[facility][1] != "consumer":
                assert(self.facilities[self.facilities[facility][1]][0] == facility)

    # Return the upstream facility 
    def get_upstream(self, src:str) -> int:
        return self.facilities[src][0]

    # Return the downstream facility 
    def get_downstream(self, src:int) -> int:
        return self.facilities[src][1]
    
    # Return the head index, which upstream is super_vendor
    def get_head(self) -> int:
        return self.head

    # Return the tail index, which downstream is consumer
    def get_tail(self) -> int:
        return self.tail

    def get_facility_count(self) -> int:
        return len(self.facilities)
