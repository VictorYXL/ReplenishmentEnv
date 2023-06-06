# ReplenishmentRL
Replenishment environment for OR and RL algorithms


## Contents

| Folder      | Description                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------- |
| [ReplenishmentEnv](ReplenishmentEnv)      | Replenishment env source code.                                      |
| [ReplenishmentEnv\config](ReplenishmentEnv\config)      | Config for building the env.                          |
| [ReplenishmentEnv\data](ReplenishmentEnv\data)      | Csv based data for skus, including sku_list, info and other dynamic data|
| [ReplenishmentEnv\env](ReplenishmentEnv\env)      | Kernel simulator for env                                    |
| [ReplenishmentEnv\utility](ReplenishmentEnv\utility)      | Utility simulator for env and wrapper               |
| [ReplenishmentEnv\wrapper](ReplenishmentEnv\wrapper)      | Wrapper for env                 |
| [Baseline](Baseline)                        | Show case of replenishment env.                                     |

## Demo to build a environment
Write config as [demo.yaml](ReplenishmentEnv\config\demo.yml)

## Run OR algorithm
```
import os
from Baseline.base_stock import BS_static, BS_dynamic
from Baseline.search_sS import sS_static, sS_hindsight
env_name = "sku200.single_store.standard"

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
print(env_name, "BS_static", sS_static_sum_balance)

# (s, S) hindsight mode
vis_path = os.path.join("output", env_name, "sS_hindsight")
sS_hindsight_sum_balance = sum(sS_hindsight(env_name, vis_path))
print(env_name, "sS_hindsight", sS_hindsight_sum_balance)
```
Visualization will be in output folder.

## Run example without installation

- Download code
```
git clone https://github.com/zhangchuheng123/ReplenishmentRL.git
```

- Build the environment

    Only python >= 3.8.0 is tested.
    - Build by conda
    ```
    conda create -n ReplenishmentEnv python=3.8.0 -f requirements.txt
    ```
    - Build by pip
    ```
    pip install -r requirements.txt
    ```

- Run example
```
    python Example\random_actions.py
```

## Build new env(Not Implement Yes)
- Prepare the data including sku_list, info and dynamic data.
- Prepare the env config following the instruction in Demo.yml
- Prepare the special or current wrapper
- Register in [ReplenishmentEnv\\__init__.py](ReplenishmentEnv\\__init__.py) file