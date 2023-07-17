import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
import torch as th
from torch.optim import RMSprop, Adam
import wandb

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer == "qplex_base":
            self.mixer = DMAQer(args)
        elif args.mixer == "qplex_alt":
            raise Exception("Not implemented here!")

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1] # a = argmax_a Q_i(s,a)
        #print("actions:", actions.shape)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t, t_env)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Pick the Q-Values for the actions taken by each agent # This is V_i
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        mac_out_maxs = mac_out.clone()
        mac_out_maxs[avail_actions == 0] = -9999999

        # Best joint-action computed by regular agents
        max_actions_qvals, max_actions_current = mac_out_maxs[:, :].max(dim=3, keepdim=True)
        max_actions_qvals = max_actions_qvals.squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t, t_env)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl
        
        if self.args.use_double_q:
            cur_max_actions = max_actions_current[:, 1:].detach()
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            #print('cur_max_actions_onehot: ', cur_max_actions_onehot.shape)
        else:
            # Best joint action computed by target agents
            target_max_qvals = target_mac_out.max(dim=3)[1]

        # Mix
        if self.args.mixer == "qplex_base":
            actions_onehot = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device)
            actions_onehot = actions_onehot.scatter(3, batch["actions"][:, :], 1)
            #print('actions_onehot: ', actions_onehot.shape)
            #print('chosen_action_qvals', chosen_action_qvals.shape)

            ans_chosen = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True) # calculate V
            #print('batch state: ', batch["state"][:, :-1].shape)
            #print('ans_chosen: ', ans_chosen.shape)
            ans_adv = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot[:, :-1],
                            max_q_i=max_actions_qvals[:, :-1], is_v=False) # calculate A
            chosen_action_qvals = ans_chosen + ans_adv # Q = A + V

            if self.args.use_double_q:
                target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                actions=cur_max_actions_onehot,
                                                max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

            # Calculate 1-step Q-Learning targets
            targets = rewards.reshape(-1,1) + self.args.gamma * (1 - terminated.reshape(-1,1)) * target_max_qvals.reshape(-1,1)
            # Td-error
            td_error = (chosen_action_qvals.reshape(-1,1) - targets)

            mask = mask.reshape(-1,1)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask
            td_loss = (masked_td_error ** 2).sum() / mask.sum()

        elif self.args.mixer == "qplex_alt":
            raise Exception("Not supported yet.")

        loss = td_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        mask_elems = mask.sum().item()
        if self.args.use_wandb:
            wandb.log({
                "td_loss": td_loss.item(),
                "grad_norm": grad_norm,
                "td_error_abs": (masked_td_error.abs().sum().item()/mask_elems),
                "td_targets": ((masked_td_error).sum().item()/mask_elems),
                "target_mean": (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                "agent_indiv_qs": ((chosen_action_qvals.reshape(-1,1) * mask).sum().item()/(mask_elems * self.args.n_agents))

            })

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def to(self,device):
        self.mac.to(device)
        self.target_mac.to(device)
        if self.mixer is not None:
            self.mixer.to(device)
            self.target_mixer.to(device)

    def save_models(self, path, postfix = ""):
        self.mac.save_models(path, postfix)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer".format(path)+postfix+".th")
        th.save(self.optimiser.state_dict(), "{}/opt".format(path)+postfix+".th")

    def load_models(self, path, postfix = ""):
        self.mac.load_models(path,postfix)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer".format(path)+postfix+".th", map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt".format(path)+postfix+".th", map_location=lambda storage, loc: storage))
