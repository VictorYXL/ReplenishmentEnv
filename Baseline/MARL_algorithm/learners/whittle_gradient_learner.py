import torch
from torch.optim import RMSprop
from torch.optim import Adam
from modules.agents import REGISTRY as a_REGISTRY
import pdb

from components.action_selectors import categorical_entropy
from components.episode_buffer import EpisodeBatch
from utils.rl_utils import build_gae_targets, build_gae_targets_with_T
from utils.value_norm import ValueNorm
import wandb


class WhittleGradientLearner(object):
    def __init__(self, w_mac, mac, scheme, logger, args):

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.w_mac = w_mac
        self.mac = mac
        self.logger = logger
        self.scheme = scheme

        self.loss_func = torch.nn.MSELoss()
        self.params = list(self.w_mac.parameters())
        if args.optim == 'RMSprop':
            self.optimiser = RMSprop(params=self.params, lr=args.w_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optim == 'Adam':
            self.optimiser = Adam(params=self.params, lr=args.w_lr)

        self.input_seq_str = (
            f"{args.actor_input_seq_str}_{self.n_agents}_{self.n_actions}"
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        w_mac_out = []
        self.w_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.w_mac.forward(batch, t, t_env)
            w_mac_out.append(agent_outs)
        whittle_indices = torch.stack(w_mac_out, dim=1)
        # B x T x Nagents x (Nactions - 1)

        tot_mac_out = []
        for i in range(self.n_actions - 1):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                lbdas = whittle_indices[:, t, :, i].detach()
                agent_outs = self.mac.forward(batch, t, t_env, lbdas=lbdas).detach()
                mac_out.append(agent_outs)
            Q = torch.stack(mac_out, dim=1)  # Concat over time
            # Q(s, <=alpha, lbda)
            Qmax_forward = torch.cummax(Q, dim=-1).values
            Qmax_forward = Qmax_forward[:, :, :, :-1]
            # Q(s, >alpha, lbda)
            Qmax_backward = torch.cummax(Q.flip(dims=(-1,)), dim=-1).values.flip(dims=(-1,))
            Qmax_backward = Qmax_backward[:, :, :, 1:]

            Q_used = (Qmax_backward - Qmax_forward)[:, :, :, i]

            tot_mac_out.append(Q_used)
        Qdiff = torch.stack(tot_mac_out, dim=-1).detach()

        loss = - (whittle_indices * Qdiff).mean()

        # Optimise agents
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.args.use_wandb:
            wandb.log({
                'whittle_loss': loss.item(),
                'whittle_avg': whittle_indices.mean().item()
            })

    def cuda(self):
        self.w_mac.cuda()

    def save_models(self, path):
        self.w_mac.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/whittle_opt.th".format(path))

    def load_models(self, path):
        self.w_mac.load_models(path)
        self.optimiser.load_state_dict(
            torch.load(
                "{}/whittle_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )