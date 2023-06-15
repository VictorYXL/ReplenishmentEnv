import torch
from torch.optim import RMSprop
from torch.optim import Adam
import copy
import pdb

from components.action_selectors import categorical_entropy
from components.episode_buffer import EpisodeBatch
from modules.critics import REGISTRY as critic_resigtry
from utils.rl_utils import build_gae_targets, build_gae_targets_with_T
from utils.value_norm import ValueNorm
import wandb


class LocalLDQNLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0

        self.params = list(self.mac.parameters())
        if args.optim == 'RMSprop':
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optim == 'Adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)

        self.target_mac = copy.deepcopy(mac)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        if self.args.use_individual_rewards:
            rewards = batch["individual_rewards"][:, :-1].to(batch.device)
        else:
            rewards = batch["reward"][:, :-1].to(batch.device)\
                .unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            if self.args.use_mean_team_reward:
                rewards /= self.n_agents
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t, t_env)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        q_values = torch.gather(mac_out[:, :-1], dim=-1, index=actions).squeeze(-1)  # Remove the last dim

        # targets and advantages
        with torch.no_grad():

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t, t_env)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # Concat across time

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=-1, keepdim=True)[1]

            if self.args.use_double_q:
                target_max_qvals = torch.gather(target_mac_out, -1, cur_max_actions).squeeze(-1)
            else:
                target_max_qvals = target_mac_out.max(dim=-1).values

            targets = rewards.squeeze(-1) + self.args.gamma * (1 - terminated) * target_max_qvals

        td_error = (q_values - targets.detach()) ** 2

        masked_td_error = td_error * mask_agent.squeeze(-1).detach()
        q_loss = (masked_td_error).sum() / mask_agent.squeeze(-1).detach().sum()

        loss = q_loss

        # Optimise agents
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = episode_num

        wandb_dict = {}
        wandb_dict.update({
            'loss': loss.item(),
            'q_values': q_values.mean().item(),
            'targets': targets.mean().item(),
            'reward': rewards.mean(),
        })
        if self.args.use_wandb:
            wandb.log(wandb_dict)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(
            torch.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )