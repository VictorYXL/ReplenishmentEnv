from ReplenishmentEnv.env.replenishment_env import ReplenishmentEnv
from ReplenishmentEnv.wrapper.default_wrapper import DefaultWrapper
import os

all = ["make_env"]
def make_env(config_name, wrapper_name="DefaultWrapper"):
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config")
    config_file = os.path.join(config_dir, config_name + ".yml")
    env = ReplenishmentEnv(config_file)

    if wrapper_name == "DefaultWrapper":
        env = DefaultWrapper(env)
    else:
        raise NotImplementedError
    return env