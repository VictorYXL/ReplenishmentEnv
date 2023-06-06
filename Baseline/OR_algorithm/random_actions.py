import random
import sys
import os
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
from ReplenishmentEnv import make_env

if __name__ == "__main__":
    env_name = "sku1000.multi_store.standard"
    exp_name = "random_action"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    env.reset()
    for i in range(10):
        action_list = [[1 * 10 for i in range(1000)] for j in range(3)]
        states, rewards, done, info_states = env.step(action_list) 
    print(info_states["balance"])