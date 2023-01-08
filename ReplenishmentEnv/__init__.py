from .env.bussiness_engineering import ReplenishmentEnv
from .wrapper.default_wrapper import DefaultWrapper
from .wrapper.dynamic_wrapper import DynamicWrapper
from .wrapper.static_wrapper import StaticWrapper
import os

all = ["make_env"]
def make_env(config_name, wrapper_name="DefaultWrapper", mode="train"):
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config")
    config_file = os.path.join(config_dir, config_name + ".yml")
    env = ReplenishmentEnv(config_file, mode)

    if wrapper_name == "DefaultWrapper":
        env = DefaultWrapper(env)
    elif wrapper_name == "DynamicWrapper":
        env = DynamicWrapper(DefaultWrapper(env))
    elif wrapper_name == "StaticWrapper":
        env = StaticWrapper(DefaultWrapper(env))
    elif wrapper_name == "ObservationWrapper":
        env = ObservationWrapper(DefaultWrapper(env))
    else:
        raise NotImplementedError
    return env