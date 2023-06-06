import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class RNNLambdaAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNLambdaAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim + args.lambda_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, lbda):

        # Batch x Agents x Dim_state
        b, a, e = inputs.size()
        inputs = inputs.view(b * a, e)

        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.view(b * a, self.args.hidden_dim)
        h_ou = self.rnn(x, h_in)
        
        lbda = lbda.unsqueeze(-1).expand(b, a, self.args.lambda_hidden_dim)
        lbda = lbda.reshape(b * a, self.args.lambda_hidden_dim)

        x = torch.cat([h_ou, lbda], dim=-1)
        q = self.fc2(x).view(b, a, -1)

        return q, h_ou
