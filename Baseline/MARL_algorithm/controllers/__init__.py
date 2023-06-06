REGISTRY = {}

from .basic_controller import BasicMAC
from .mappo_controller import MAPPOMAC
from .dqn_controller import DQNMAC
from .ldqn_controller import LDQNMAC
from .whittle_disc_controller import WhittleDiscreteMAC
from .whittle_cont_controller import WhittleContinuousMAC
from .mappo_controller_with_base_stock import MAPPOMACWITHBASESTOCK


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mappo_mac"] = MAPPOMAC
REGISTRY["dqn_mac"] = DQNMAC
REGISTRY["ldqn_mac"] = LDQNMAC
REGISTRY["whittle_disc_mac"] = WhittleDiscreteMAC
REGISTRY["whittle_cont_mac"] = WhittleContinuousMAC
REGISTRY["mappo_mac_with_base_stock"] = MAPPOMACWITHBASESTOCK
