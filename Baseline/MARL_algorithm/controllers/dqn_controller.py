import torch
import random
import pdb

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from utils.input_utils import build_actor_inputs, get_actor_input_shape


# This multi-agent controller shares parameters between agents
class DQNMAC:
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

    def select_actions(self, ep_batch, t_ep, t_env, lbda_indices, bs=slice(None), test_mode=False):

        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)

        if self.args.use_n_lambda:
            agent_outputs = agent_outputs.reshape(agent_outputs.shape[0], agent_outputs.shape[1], self.args.n_lambda, -1)
            # agent_outputs.shape = (batch_size, #SKU, #lambda, #action)
            gather_indices = lbda_indices.view(self.args.batch_size, 1, 1, 1)\
                .expand(self.args.batch_size, agent_outputs.shape[1], 1, agent_outputs.shape[-1])
            agent_outputs = torch.gather(agent_outputs, dim=2, index=gather_indices).squeeze(2)

        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, t_env, test_mode=False):
        agent_inputs = build_actor_inputs(self.input_seq_str, ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(
                batch_size, self.n_agents, -1
            )  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path, postfix=""):
        torch.save(self.agent.state_dict(), "{}/agent".format(path)+postfix+".th")

    def load_models(self, path,postfix=""):
        self.agent.load_state_dict(
            torch.load(
                "{}/agent".format(path)+postfix+".th", map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                torch.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape