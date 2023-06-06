import torch

def build_critic_inputs(inputs_seq_str, batch, t=None):
    bs = batch.batch_size
    max_t = batch.max_seq_length if t is None else 1
    ts = slice(None) if t is None else slice(t, t + 1)
    inputs = []
    n_agents = int(inputs_seq_str.split("_")[-3])
    n_actions = int(inputs_seq_str.split("_")[-2])
    share_critic = int(inputs_seq_str.split("_")[-1])
    if 's' in inputs_seq_str:
        n_agents = 1
    
    for item in inputs_seq_str.split("_")[:-2]:
        if item == "s":
            # state
            state = batch["state"][:, ts]
            inputs.append(state.unsqueeze(2).repeat(1, 1, n_agents, 1))
        elif item == "g":
            state = batch["obs"][:, ts, :, 58: 58 + 28] # 28 dim global state
            inputs.append(state)
        elif item == "o":
            # observation
            inputs.append(batch["obs"][:, ts])
        elif item == "a":
            # actions (masked out by agent)
            actions = (
                batch["actions_onehot"][:, ts]
                .view(bs, max_t, 1, -1)
                .repeat(1, 1, n_agents, 1)
            )
            agent_mask = 1 - torch.eye(n_agents, device=batch.device)
            agent_mask = agent_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)
            inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        elif item == "la":
            # last actions
            if t == 0:
                inputs.append(
                    torch.zeros_like(batch["actions_onehot"][:, 0:1])
                    .view(bs, max_t, 1, -1)
                    .repeat(1, 1, n_agents, 1)
                )
            elif isinstance(t, int):
                inputs.append(
                    batch["actions_onehot"][:, slice(t - 1, t)]
                    .view(bs, max_t, 1, -1)
                    .repeat(1, 1, n_agents, 1)
                )
            else:
                last_actions = torch.cat(
                    [
                        torch.zeros_like(batch["actions_onehot"][:, 0:1]),
                        batch["actions_onehot"][:, :-1],
                    ],
                    dim=1,
                )
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(
                    1, 1, n_agents, 1
                )
                inputs.append(last_actions)
        elif item == "i":
            inputs.append(
                torch.eye(n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, max_t, -1, -1)
            )
        elif item == "la^i":
            # last individual actions
            inputs.append(batch["actions_onehot"][:, ts])

    
    inputs = torch.cat([x.reshape(bs, max_t, n_agents, -1) for x in inputs], dim=-1)
    if share_critic:
        inputs = inputs[:, :, 0:1]

    return inputs, bs, max_t


def get_critic_input_shape(inputs_seq_str, scheme):
    input_shape = 0
    n_agents = int(inputs_seq_str.split("_")[-3])
    n_actions = int(inputs_seq_str.split("_")[-2])
    share_critic = int(inputs_seq_str.split("_")[-1])
    assert n_actions == scheme["actions_onehot"]["vshape"][0]
    # for traditional inputs
    for item in inputs_seq_str.split("_")[:-2]:
        if item == "s":
            # state
            input_shape += scheme["state"]["vshape"]
        elif item == "g":
            input_shape += 28
        elif item == "o":
            # observation
            input_shape += scheme["obs"]["vshape"]
        elif item == "a":
            # actions
            input_shape += n_actions * n_agents
        elif item == "la":
            # last action
            input_shape += n_actions * n_agents
        elif item == "i":
            # agent id
            input_shape += n_agents
        elif item == "la^i":
            input_shape += n_actions


    return input_shape


def build_actor_inputs(inputs_seq_str, batch, t=None):
    bs = batch.batch_size
    inputs = []
    n_agents = int(inputs_seq_str.split("_")[-2])

    for item in inputs_seq_str.split("_")[:-2]:
        if item == "s":
            # state
            inputs.append(batch["state"][:, t])
        elif item == "o":
            # observation
            inputs.append(batch["obs"][:, t])
        elif item == "la":
            # last actions
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        elif item == "i":
            inputs.append(
                torch.eye(n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

    inputs = torch.cat([x.reshape(bs, n_agents, -1) for x in inputs], dim=-1)
    return inputs


def get_actor_input_shape(inputs_seq_str, scheme):
    input_shape = 0
    n_agents = int(inputs_seq_str.split("_")[-2])
    n_actions = int(inputs_seq_str.split("_")[-1])
    assert n_actions == scheme["actions_onehot"]["vshape"][0]
    for item in inputs_seq_str.split("_")[:-2]:
        if item == "s":
            # state
            input_shape += scheme["state"]["vshape"]
        elif item == "o":
            # observation
            input_shape += scheme["obs"]["vshape"]
        elif item == "la":
            # last action
            input_shape += n_actions
        elif item == "i":
            # agent id
            input_shape += n_agents
    return input_shape