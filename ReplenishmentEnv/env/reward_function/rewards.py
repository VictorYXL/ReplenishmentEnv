from typing import Tuple
import numpy as np

from ReplenishmentEnv.env.agent_states import AgentStates

"""
    reward1: calculate the cost when buy in.
    agent_states: object for AgentStates to save all agents info in current env step.
    profit_info: env state. For reward1, following keys are needed:
        - backlog_ratio: backlog = (selling_price - procurement_cost) * (cur_demand - sale) * backlog_ratio
"""
def reward1(agent_states: AgentStates, profit_info: dict) -> Tuple[np.array, dict]:
    selling_price      = agent_states["selling_price"]
    procurement_cost   = agent_states["procurement_cost"]
    sale               = agent_states["sale"]
    replenish          = agent_states["replenish"]
    order_cost         = agent_states["order_cost"]
    in_stocks          = agent_states["in_stock"]
    storage_cost       = agent_states["storage_cost"]
    holding_cost_ratio = agent_states["holding_cost_ratio"]
    current_demand     = agent_states["demand"]
    backlog_ratio      = profit_info.get("backlog_ratio", 0)

    # TODO: discuss whether to add (1 - excess_ratio) * excess as compensation
    income       = selling_price * sale
    outcome      = procurement_cost * replenish
    order_cost   = order_cost * np.where(replenish > 0, 1, 0)
    holding_cost = (storage_cost + selling_price * holding_cost_ratio) * in_stocks
    backlog      = (selling_price - procurement_cost) * (current_demand - sale) * backlog_ratio

    reward       = income - outcome - order_cost - holding_cost - backlog
    reward_info = {
        "income":       income,
        "outcome":      outcome,
        "order_cost":   order_cost,
        "holding_cost": holding_cost,
        "backlog":      backlog,
    }

    return reward, reward_info

"""
    reward2: calculate the cost when sale.
    agent_states: object for AgentStates to save all agents info in current env step.
    profit_info: env state. For reward2, following keys are needed:
        - backlog_ratio: backlog = (selling_price - procurement_cost) * (cur_demand - sale) * backlog_ratio
        - excess_ratio: Only excess_ratio of excess skus will cost loss in profit
"""
def reward2(agent_states: AgentStates, profit_info: dict) -> Tuple[np.array, dict]:
    selling_price      = agent_states["selling_price"]
    procurement_cost   = agent_states["procurement_cost"]
    sale               = agent_states["sale"]
    excess             = agent_states["excess"]
    replenish          = agent_states["replenish"]
    order_cost         = agent_states["order_cost"]
    in_stocks          = agent_states["in_stock"]
    current_demand     = agent_states["demand"]
    storage_cost       = agent_states["storage_cost"]
    holding_cost_ratio = agent_states["holding_cost_ratio"]
    backlog_ratio      = profit_info.get("backlog_ratio", 0)
    excess_ratio       = profit_info.get("excess_ratio", 1)

    profit       = sale * (selling_price - procurement_cost)
    excess       = procurement_cost * excess * excess_ratio
    order_cost   = order_cost * np.where(replenish > 0, 1, 0)
    holding_cost = (storage_cost + selling_price * holding_cost_ratio) * in_stocks
    backlog      = (selling_price - procurement_cost) * (current_demand - sale) * backlog_ratio
    backlog      = np.where(backlog > 0, backlog, 0)

    reward      = profit - excess - order_cost - holding_cost - backlog
    reward_info = {
        "profit":       profit,
        "excess":       excess,
        "order_cost":   order_cost,
        "holding_cost": holding_cost,
        "backlog":      backlog,
    }

    return reward, reward_info
