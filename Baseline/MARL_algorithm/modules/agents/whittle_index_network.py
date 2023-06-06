import torch.nn as nn
import torch.nn.functional as F
import pdb


class WhittleIndexNetwork(nn.Module):
    def __init__(self, input_shape, args):
        super(WhittleIndexNetwork, self).__init__()

        self.args = args
        self.fc1 = nn.Linear(input_shape, args.whittle_hidden_dim)
        self.rnn = nn.GRUCell(args.whittle_hidden_dim, args.whittle_hidden_dim)
        self.fc3 = nn.Linear(args.whittle_hidden_dim, args.n_actions - 1)

    def forward(self, inputs, hidden_states):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_hidden(self):
        # In this way, we can make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.whittle_hidden_dim).zero_()

    def forward(self, inputs, hidden_states):

        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        hidden_states = hidden_states.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, hidden_states)
        q = self.fc3(hh)
        q = q + self.args.w_agent_offset

        return q.view(b, a, -1), hh.view(b, a, -1)
