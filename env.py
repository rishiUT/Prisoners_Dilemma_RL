import agent
from agent import *
import numpy as np
from statevars import STATE_VARS
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple, Box
from itertools import combinations
from math import comb

class PDGame(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        super().__init__()
        self.action_space = Discrete(2)
        self.state = None
        self.num_agents = 3

        self._agent_ids = {idx for idx in range(self.NUM_AGENTS)}
        self.state_size = len(list(STATE_VARS))
        self.observation_space = MultiDiscrete([2]*self.state_size)
        self.iterations = 10
        self.gen = combinations(range(self.num_agents),2)
        self.num_matches = comb(self.num_agents,2)
        self.num_match = 0

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def reset(self):
        self.state = np.zeros((self.num_agents,self.state_size))
        self.gen = combinations(range(self.num_agents),2)
        self.num_match = 0
        return self._obs()

    def step(self, action_dict):
        match_players = next(self.gen)
        a1_act = action_dict[match_players[0]]
        a2_act = action_dict[match_players[1]]
        self.num_match += 1
        if self.num_match == self.num_agents:
            done = True
        rewards = {match_players[0] : self.get_reward(a1_act,a2_act), match_players[1]: self.get_reward(a2_act,a1_act)}
        obs = self._obs()
        dones = {"__all__": done}
        infos = {}
        return obs, rewards, dones, infos

    def _obs(self):
        return {
            agent_id : {"obs": self.agent_obs(agent_id)} for agent_id in range(self.NUM_AGENTS)
        }

    def agent_obs(self, agent_id):
        return self.state
    def get_reward(self, agent_act, opponent_act):
        # There is a simpler way to implement this if defect and cooperate are 0 and 1,
        # But this method should stay accurate if we decide to change how we represent defect and cooperate
        if agent_act == ACTIONS.DEFECT:
            if opponent_act == ACTIONS.COOPERATE:
                return 4
            else:
                return 1
        else:
            if opponent_act == ACTIONS.COOPERATE:
                return 3
            else:
                return 0


class Environment():
    def __init__(self,agents) -> None:
        state_size = len(list(STATE_VARS))
        
        
        self.agents = agents
        if not self.agents:
            self.agents.append(Cynic(state_size,0))
            self.agents.append(EasyMark(state_size,1))
            self.agents.append(TitForTat(state_size,2))
        print(f"Agents: {self.agents}")
        self.total_reward = 0
        num_agents = len(agents)
        self.last_acts = np.zeros((num_agents, num_agents),dtype=int)
        self.state = np.zeros((num_agents,state_size))



    def run_sim(self) -> None:
        for i in range(10):
            for agent_idx, agent in enumerate(self.agents):
                for opp_idx, opponent in enumerate(self.agents):
                    if opponent is not agent:
                        print(f"Agent {agent_idx}:")
                        agent_action = agent.act(self.get_state(agent, opponent))
                        print(f"Opp {opp_idx}:")
                        opp_action = opponent.act(self.get_state(opponent, agent))
                        agent_reward = self.get_reward(agent_action, opp_action)
                        opp_reward = self.get_reward(opp_action, agent_action)
                        print(f"Agent {agent_idx}:")
                        agent.get_reward(agent_reward)
                        print(f"Opp {opp_idx}:")
                        opponent.get_reward(opp_reward)
                        self.total_reward += agent_reward + opp_reward
                        self.last_acts[agent_idx][opp_idx] = agent_action
                        self.last_acts[opp_idx][agent_idx] = opp_action
                total_rewards = [agent.reward_total for agent in self.agents]
                total_rewards_arr = np.array(total_rewards)
                best_agent_idx = np.argmax(total_rewards_arr)
                print(f"Agent {best_agent_idx}/ {str(self.agents[best_agent_idx])} wins")

    def get_state(self, agent, opponent):
        #This is incomplete, once we have state variables we can determine how they're determined
        self.state[agent.id,STATE_VARS.OPP_LAST_ACT] = self.last_acts[opponent.id][agent.id]
        return self.state[agent.id]

    def get_reward(self, agent_act, opponent_act):
        # There is a simpler way to implement this if defect and cooperate are 0 and 1,
        # But this method should stay accurate if we decide to change how we represent defect and cooperate
        if agent_act == ACTIONS.DEFECT:
            if opponent_act == ACTIONS.COOPERATE:
                return 4
            else:
                return 1
        else:
            if opponent_act == ACTIONS.COOPERATE:
                return 3
            else:
                return 0