import numpy as np

def equal_accept(arrived, capacity, agent_states, warehouse):
    total_arrived = sum(arrived)

    # Calculate accept ratio due to the capacity limitation.
    in_stock_volume = np.sum(agent_states[warehouse, "in_stock"] * agent_states[warehouse, "volume"])
    remaining_space = capacity - in_stock_volume
    accept_ratio = min(remaining_space / total_arrived, 1.0) if total_arrived > 0 else 0

    # Receive skus by accept ratio
    accept_amount = arrived * accept_ratio

    return accept_amount