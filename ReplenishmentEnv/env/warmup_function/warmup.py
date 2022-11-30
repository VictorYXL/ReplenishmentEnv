"""
    Warm up the env before action in look back length.
    In warm up process, replenish quantity will be set to last day's demand.
"""
def replenish_by_last_demand(env) -> None:
    _init_balance = env.balance
    _action_mode = env.action_mode

    env.action_mode = "continuous"
    last_demand = [0] * len(env.sku_list)    
    for day in range(env.lookback_len):
        env.step(last_demand)
        last_demand = env.agent_states["demand", "history"][-1]

    env.action_mode = _action_mode
    env.balance = _init_balance
