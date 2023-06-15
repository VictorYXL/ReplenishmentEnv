import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.input_utils import build_critic_inputs, get_critic_input_shape
from utils.th_utils import orthogonal_init_


class MAPPORNNCriticShare(nn.Module):
    def __init__(self, scheme, args):
        super(MAPPORNNCriticShare, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = int(args.n_agents/3)
        self.input_seq_str = (
            f"{args.critic_input_seq_str}_{self.n_agents}_{self.n_actions}_{1}"
        )

        input_shape = get_critic_input_shape(self.input_seq_str, scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

        self.hidden_states = None

    def init_hidden(self, batch_size):
        hidden_states = self.fc1.weight.new(1, self.args.hidden_dim).zero_()
        self.hidden_states = hidden_states.unsqueeze(0).expand(
            batch_size, 1, -1
        )  # bav

    def forward(self, batch, t):
        inputs, bs, max_t = build_critic_inputs(self.input_seq_str, batch, t=t)
        x = F.relu(self.fc1(inputs)).view(-1, self.args.hidden_dim)
        h_in = self.hidden_states.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)
        self.hidden_states = hh.view(bs, 1, -1)
        return q.view(bs, 1, -1).expand(bs, self.n_agents, 1)