import torch
import random
import pdb

from modules.agents.whittle_index_network import WhittleIndexNetwork
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from utils.input_utils import build_actor_inputs, get_actor_input_shape


# This multi-agent controller shares parameters between agents
class WhittleDiscreteMAC(object):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.input_seq_str = (
            f"{args.actor_input_seq_str}_{self.n_agents}_{self.n_actions}"
        )

        input_shape = get_actor_input_shape(self.input_seq_str, scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def _build_agents(self, input_shape):

        self.agent = agent_REGISTRY[self.args.w_agent](input_shape, self.args)
        # self.agent = WhittleIndexNetwork(input_shape, self.args)

    def forward(self, ep_batch, t_ep, t_env, test_mode=False):

        agent_inputs = build_actor_inputs(self.input_seq_str, ep_batch, t_ep)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def select_actions(self, ep_batch, t_ep, t_env, lbda_indices, bs=slice(None), test_mode=False):

        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)

        lbda_indices = lbda_indices.unsqueeze(1).unsqueeze(2).expand_as(agent_outputs)
        chosen_actions = (agent_outputs > lbda_indices).sum(-1)

        return chosen_actions

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0)\
                .expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/whittle_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            torch.load(
                "{}/whittle_agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )
