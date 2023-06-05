from Example.base_stock import *
from Example.search_Ss import *

if __name__ == "__main__":
    env_names = [
        "sku200.single_store.standard"

    ]
    for env_name in env_names:
        # bsp_oracle(env_name)
        # bsp_dynamic_history(env_name)
        # cSs_static_independent(env_name)
        # bsp_static(env_name)
        # Ss_oracle_independent(env_name)
        # bsp_static(env_name)
        Ss_oracle_independent(env_name)
        # bsp_dynamic_21(env_name)