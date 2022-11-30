import random
import gym
import sys
import os
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
import ReplenishmentEnv

if __name__ == "__main__":
    env = gym.make("sku58-v0")
    env.reset()
    for i in range(10):
        action_list = [int(random.random() * 10) for i in range(58)]
        states, rewards, dones, info_states = env.step(action_list) 
    print(info_states["balance"])