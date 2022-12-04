import random
import sys
import os
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
from ReplenishmentEnv import make_env

if __name__ == "__main__":
    env = make_env("sku58", wrapper_name = "DefaultWrapper", mode="test")
    env.reset()
    for i in range(10):
        action_list = [int(random.random() * 10) for i in range(58)]
        states, rewards, done, info_states = env.step(action_list) 
    print(info_states["balance"])