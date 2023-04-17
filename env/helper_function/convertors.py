import numpy as np

def continuous(actions, action_config, agent_states):
    return actions

def discrete(actions, action_config, agent_states):
    assert("space" in action_config)
    action_space = np.array(action_config["space"])
    return np.round(action_space[actions])

def demand_mean_continuous(actions, action_config, agent_states):
    history_demand_mean = np.average(agent_states["all_warehouses", "demand", "lookback"], 1)
    return actions * history_demand_mean

def demand_mean_discrete(actions, action_config, agent_states):
    history_demand_mean = np.average(agent_states["all_warehouses", "demand", "lookback"], 1)
    assert("space" in action_config)
    action_space = np.array(action_config["space"])
    return action_space[actions] * history_demand_mean
