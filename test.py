import os
from Example.base_stock import BS_static, BS_dynamic
from Example.search_sS import sS_static, sS_hindsight
env_name = "sku200.single_store.add_gap_1"

# Base stock static mode
vis_path = os.path.join("output", env_name, "BS_static")
BS_static_sum_balance = sum(BS_static(env_name, vis_path))
print(env_name, "BS_static", BS_static_sum_balance)

# Base stock dynamic mode
vis_path = os.path.join("output", env_name, "BS_dynamic")
BS_static_sum_balance = sum(BS_dynamic(env_name, vis_path))
print(env_name, "BS_dynamic", BS_static_sum_balance)

# (s, S) static mode
vis_path = os.path.join("output", env_name, "sS_static")
sS_static_sum_balance = sum(sS_static(env_name, vis_path))
print(env_name, "sS_static", sS_static_sum_balance)

# (s, S) hindsight mode
vis_path = os.path.join("output", env_name, "sS_hindsight")
sS_hindsight_sum_balance = sum(sS_hindsight(env_name, vis_path))
print(env_name, "sS_hindsight", sS_hindsight_sum_balance)