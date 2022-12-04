from ReplenishmentEnv.env.supply_chain.facility import Facility

"""
    SupplyChain class, contains a lift of facilities.
    Currently, only chainlike supply chain are supprted. 
    Each facility has only 1 upstream and 1 downstream.
    Head facility receive skus from super_vendor and tail facility sells skus to consumer.
"""
class SupplyChain:
    def __init__(self, facility_config) -> None:
        """
            self.facilities = [
                [upstream_index, downstream_index],
                [upstream_index, downstream_index],
                ...
            ]
            Index represents the order as actions, rewards, and others in env.
            Before the supply chain is built, index is represented by facility name.
        """
        self.facilities = []
        self.name_to_index = {}
        self.head = None
        self.tail = None
        for index, facility in enumerate(facility_config):
            assert("name" in facility)
            assert("upstream" in facility)
            assert("downstream" in facility)
            assert("sku" in facility)
            self.name_to_index[facility["name"]] = index
            if facility["upstream"] == "super_vendor":
                self.head = index
                facility["upstream"] = -1
            if facility["downstream"] == "consumer":
                self.tail = index
                facility["downstream"] = -2
            self.facilities.append([facility["upstream"], facility["downstream"]])
        for facility in self.facilities:
            if isinstance(facility[0], str):
                facility[0] = self.name_to_index[facility[0]]
            if isinstance(facility[1], str):
                facility[1] = self.name_to_index[facility[1]]

        assert(self.head is not None)
        assert(self.tail is not None)
        for index, facility in enumerate(self.facilities):
            assert(isinstance(facility[0], int))
            assert(isinstance(facility[1], int))
            if facility[0] >= 0:
                assert(self.facilities[facility[0]][1] == index)
            if facility[1] >= 0:
                assert(self.facilities[facility[1]][0] == index)
        pass
            

    
    def build_data(self) -> None:
        pass
