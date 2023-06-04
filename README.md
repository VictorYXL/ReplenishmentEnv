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
| [ReplenishmentEnv\wrapper](ReplenishmentEnv\wrapper)      | Wrapper for env(Not implement yet)c                 |
| [Example](Example)                        | Show case of replenishment env.                                     |

## Install from Source

- Download code
```
git clone https://github.com/zhangchuheng123/ReplenishmentRL.git
```

- Build and install 
```
python setup.py install
```

- Run example
```
import random
import os
from ReplenishmentEnv import make_env

env_name = "sku200.single_store.standard"
vis_path = os.path.join("output", "random_action")
env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
env.reset()
for i in range(10):
    action_list = [[random.random() for i in range(200)]]
    states, rewards, done, info_states = env.step(action_list) 
print(info_states["balance"])
```

## Run example without installation

- Download code
```
git clone https://github.com/VictorYXL/ReplenishmentEnv.git
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