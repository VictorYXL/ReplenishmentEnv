import os

from ReplenishmentEnv.env.replenishment_env import ReplenishmentEnv
from ReplenishmentEnv.wrapper.default_wrapper import DefaultWrapper
from ReplenishmentEnv.wrapper.dynamic_wrapper import DynamicWrapper
from ReplenishmentEnv.wrapper.history_wrapper import HistoryWrapper
from ReplenishmentEnv.wrapper.observation_wrapper import ObservationWrapper
from ReplenishmentEnv.wrapper.observation_wrapper_for_old_code import ObservationWrapper4OldCode
from ReplenishmentEnv.wrapper.flatten_wrapper import FlattenWrapper
from ReplenishmentEnv.wrapper.oracle_wrapper import OracleWrapper


all = ["make_env"]


def make_env(config_name, wrapper_names=["DefaultWrapper"], mode="train", vis_path=None, update_config=None, config_dir=None):
    if config_dir is None:
        config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config")
    config_file = os.path.join(config_dir, config_name + ".yml")
    env = ReplenishmentEnv(config_file, mode, vis_path, update_config)

    for wrapper_name in wrapper_names:
        env = eval(wrapper_name)(env)
    return env