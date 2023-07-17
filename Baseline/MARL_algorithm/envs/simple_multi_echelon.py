import numpy as np
class SimpleMultiEchelonEnv(object):
    def __init__(self, 
    n_agents = 100,
        task_type = "Standard",
        mode = "train",
        time_limit=1460,
        vis_path=None,
        echelon_num = 5, 
        embedding_dim = 6, 
        seed = 101,
        **kwargs):
    
        self.n_warehouses = echelon_num
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.episode_limit = time_limit
        np.random.seed(seed)
        embedding_matrix = np.random.randn(echelon_num, embedding_dim)
        self.agent_embedding = np.eye(echelon_num).dot(embedding_matrix)
        self.in_stock = np.zeros(echelon_num)

        self.holding_cost = 5
        self.backlog = 5
        # self.selling_profit = 10
        self.selling_profit = 5*self.n_warehouses
    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = actions.reshape(-1)
        self.env_t += 1
        total_return = 0
        # actions = [int(a) for a in actions]
        for i in range(len(actions)):
            if i == 0:
                if actions[i] == 1:
                    self.in_stock[0] = 1

            elif i > 0 and i < len(actions) - 1:
                if actions[i] == 1:
                    if self.in_stock[i-1] == 1:
                        self.in_stock[i-1] = 0
                        self.in_stock[i] = 1
                        # total_return += self.selling_profit
                    elif self.in_stock[i-1] == 0:
                        total_return -= self.backlog

            elif i == len(actions) - 1:
                if actions[i] == 1:
                    if self.in_stock[i-1] == 1:
                        # 已经是最后一层了，in_stock[i]不需要再设置为1了。进货了也会立马买
                        self.in_stock[i-1] = 0
                        self.in_stock[i] = 1
                        # total_return += self.selling_profit
                    elif self.in_stock[i-1] == 0:
                        total_return -= self.backlog
                if self.in_stock[i] == 1:
                    total_return += self.selling_profit
                    self.in_stock[i] = 0
                else:
                    # 已经是最后一层了，customer一定有需求，所以立马得到一个backlog
                    total_return -= self.backlog
        total_return -= self.holding_cost * self.in_stock.sum()
        return total_return, True, None

        

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.agent_embedding

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.agent_embedding[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.embedding_dim

    def get_state(self):
        return self.agent_embedding.reshape(-1)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_warehouses * self.embedding_dim

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_warehouses):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        # only action 0 and 1
        return np.ones(2)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # only action 0 and 1
        return 2

    def reset(self):
        """ Returns initial observations and states"""
        self.in_stock = np.zeros(self.n_warehouses)
        self.env_t = 0
        # return obs and state
        return self.agent_embedding, self.agent_embedding.reshape(-1)
    
    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        return self.seed

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_warehouses,
                    "episode_limit": self.episode_limit,
                    "n_warehouses": self.n_warehouses}
        return env_info
