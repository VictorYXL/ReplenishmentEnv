REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_lambda_agent import RNNLambdaAgent
from .whittle_index_network import WhittleIndexNetwork

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_lambda"] = RNNLambdaAgent
REGISTRY["whittle_index_network"] = WhittleIndexNetwork