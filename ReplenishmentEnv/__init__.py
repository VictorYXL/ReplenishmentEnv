from .env.replenishment_env import ReplenishmentEnv
from .wrapper.default_wrapper import DefaultWrapper
from .wrapper.dynamic_wrapper import DynamicWrapper
from .wrapper.static_wrapper import StaticWrapper
from .wrapper.observation_wrapper import ObservationWrapper
from .wrapper.observation_wrapper_for_old_code import ObservationWrapper4OldCode
from .wrapper.flatten_wrapper import FlattenWrapper
import os

all = ["make_env"]


def make_env(config_name, wrapper_names=["DefaultWrapper"], mode="train"):
    config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config")
    config_file = os.path.join(config_dir, config_name + ".yml")
    env = ReplenishmentEnv(config_file, mode)

    for wrapper_name in wrapper_names:
        env = eval(wrapper_name)(env)
    return env