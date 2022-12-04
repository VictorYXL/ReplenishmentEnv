from ReplenishmentEnv.env.replenishment_env import ReplenishmentEnv
from ReplenishmentEnv.wrapper.default_wrapper import DefaultWrapper
from ReplenishmentEnv.wrapper.oracle_wrapper import OracleWrapper
import os

all = ["make_env"]
def make_env(config_name, wrapper_name="DefaultWrapper", mode="train"):
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config")
    config_file = os.path.join(config_dir, config_name + ".yml")
    env = ReplenishmentEnv(config_file, mode)

    if wrapper_name == "DefaultWrapper":
        env = DefaultWrapper(env)
    elif wrapper_name == "OracleWrapper":
        env = OracleWrapper(DefaultWrapper(env))
    else:
        raise NotImplementedError
    return env