import random
import sys
import os
env_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")
sys.path.insert(0, env_dir)
from ReplenishmentEnv import make_env
def get_ranks(my_array):
    # my_array = np.array([7, 2, 5, 1, 9])

    # 使用argsort()函数获取元素排序后的索引数组
    sorted_indices = np.argsort(my_array)

    # 创建一个与原始数组相同大小的数组，用于存储每个元素的排名
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(my_array)) + 1
    return ranks
if __name__ == "__main__":
    env_name = "sku1000.multi_store.standard"
    exp_name = "random_action"
    vis_path = os.path.join("output", exp_name)
    env = make_env(env_name, wrapper_names=["DefaultWrapper"], mode="test", vis_path=vis_path)
    env.reset()
    for i in range(10):
        action_list = [[random.random() * 10 for i in range(1000)] for j in range(3)]
        states, rewards, done, info_states = env.step(action_list) 
    print(info_states["balance"])