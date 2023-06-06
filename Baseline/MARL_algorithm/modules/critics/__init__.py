from .coma import COMACritic
from .mappo_rnn_critic import MAPPORNNCritic
from .mappo_rnn_critic_share import MAPPORNNCriticShare

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["mappo_rnn_critic"] = MAPPORNNCritic
REGISTRY["mappo_rnn_critic_share"] = MAPPORNNCriticShare
