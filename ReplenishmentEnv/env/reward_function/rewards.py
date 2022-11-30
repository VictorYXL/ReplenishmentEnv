import numpy as np

from ReplenishmentEnv.env.agent_states import AgentStates

"""
    reward1: calculate the cost when buy in.
    agent_states: object for AgentStates to save all agents info in current env step.
    env_info: env state. For reward1, following keys are needed:
        - cur_demand: oracle current demand for all skus
        - unit_storage_cost: unified unit storage cost for all skus
        - backlog_ratio: backlog = (selling_price - procurement_cost) * (cur_demand - sale) * backlog_ratio
"""
def reward1(agent_states: AgentStates, env_info: dict) -> np.array:
    selling_price       = agent_states["selling_price"]
    procurement_cost    = agent_states["procurement_cost"]
    sale                = agent_states["sale"]
    replenish           = agent_states["replenish"]
    order_cost          = agent_states["order_cost"]
    in_stocks           = agent_states["in_stock"]
    volume              = agent_states["volume"]
    unit_storage_cost   = env_info["unit_storage_cost"]
    current_demand      = env_info["current_demand"]
    backlog_ratio       = env_info["backlog_ratio"]

    income      = selling_price * sale
    outcome     = procurement_cost * replenish
    order_cost  = order_cost * np.where(replenish > 0, 1, 0)
    rent        = unit_storage_cost * in_stocks * volume
    backlog     = (selling_price - procurement_cost) * (current_demand - sale) * backlog_ratio

    reward      = income - outcome - order_cost - rent - backlog
    return reward

"""
    reward2: calculate the cost when sale.
    agent_states: object for AgentStates to save all agents info in current env step.
    env_info: env state. For reward2, following keys are needed:
        - cur_demand: oracle current demand for all skus
        - unit_storage_cost: unified unit storage cost for all skus
        - backlog_ratio: backlog = (selling_price - procurement_cost) * (cur_demand - sale) * backlog_ratio
"""
def reward2(agent_states: AgentStates, env_info: dict) -> np.array:
    selling_price       = agent_states["selling_price"]
    procurement_cost    = agent_states["procurement_cost"]
    sale                = agent_states["sale"]
    excess              = agent_states["excess"]
    replenish           = agent_states["replenish"]
    order_cost          = agent_states["order_cost"]
    in_stocks           = agent_states["in_stock"]
    volume              = agent_states["volume"]
    unit_storage_cost   = env_info["unit_storage_cost"]
    current_demand      = env_info["current_demand"]
    backlog_ratio       = env_info["backlog_ratio"]

    profit      = sale * (selling_price - procurement_cost)
    excess      = procurement_cost * excess
    order_cost  = order_cost * np.where(replenish > 0, 1, 0)
    rent        = unit_storage_cost * in_stocks * volume
    backlog     = (selling_price - procurement_cost) * (current_demand - sale) * backlog_ratio

    reward      = profit - excess - order_cost - rent - backlog
    return reward
