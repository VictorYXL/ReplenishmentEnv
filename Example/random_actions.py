import random
import sys
import os
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
from ReplenishmentEnv import make_env

if __name__ == "__main__":
    env = make_env("sku50.MultiStore.Standard", wrapper_names=["DefaultWrapper"], mode="test")
    env.reset(os.path.join("output", "random_action"))
    for i in range(10):
        action_list = [[int(random.random() * 10) for i in range(50)] for j in range(3)]
        states, rewards, done, info_states = env.step(action_list) 
    print(info_states["balance"])