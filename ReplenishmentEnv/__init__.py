from gym.envs.registration import register
import os
config_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config")
for config_name in os.listdir(config_dir):
    env_name = os.path.splitext(config_name)[0]
    register(
        id="{0}-v0".format(env_name),
        entry_point="ReplenishmentEnv.env:ReplenishmentEnv",
        disable_env_checker=True,
        kwargs={"config_path": os.path.join(config_dir, config_name)}
    )
