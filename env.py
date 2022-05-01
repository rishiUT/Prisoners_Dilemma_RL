from argparse import Action
import os

from torch import randint
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import agent
from agent import *
import numpy as np
from statevars import STATE_VARS, ACTIONS
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from gym.spaces import Dict, Discrete, MultiDiscrete, Tuple, Box
from itertools import combinations
import random

class PDGame(MultiAgentEnv):
    action_space = Discrete(2)

    def __init__(self, env_config):
        super().__init__()
        self.action_space = Discrete(2)
        self.state = None
        self.num_agents = 2
        self.rounds = random.randint(2, 10)

        self._agent_ids = {idx for idx in range(self.num_agents)}
        self.state_size = len(list(STATE_VARS))
        self.observation_space = Box(low=float('-inf'), high=float('inf'), shape=(self.state_size,), dtype=np.float32)
        # print("AAA ", self.observation_space)
        
        self.gen = combinations(range(self.num_agents),2)
        self.num_matches = len(list(combinations(range(self.num_agents),2)))
        self.next_match = next(self.gen)
        self.num_match = 0
        self.last_acts = np.zeros((self.num_agents, self.num_agents),dtype=int)
        self.turns_since_defect = np.zeros((self.num_agents), dtype=int)
        self.agents_num_rounds = np.zeros((self.num_agents), dtype=int)
        self.total_score = 0
        self.agent_scores = np.zeros((self.num_agents), dtype=int)
        self.num_defections = np.zeros((self.num_agents), dtype=int)
        self.state = np.zeros((self.state_size))
        self.curr_round = 0

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def action_space_sample(self, agent_ids: list = None):
        agent_id_0 = self.next_match[0]
        agent_id_1 = self.next_match[1]
        # 
        ret = {
            agent_id_0 : 0,
            agent_id_1 : 0,
        }
        return ret

    def reset(self):
        # print("Resetting env")
        self.state = np.zeros((self.state_size))
        self.gen = combinations(range(self.num_agents),2)
        self.num_match = 0
        self.last_acts = np.zeros((self.num_agents, self.num_agents),dtype=int)
        self.turns_since_defect = np.zeros((self.num_agents), dtype=int)
        self.agents_num_rounds = np.zeros((self.num_agents), dtype=int)
        self.total_score = 0
        self.agent_scores = np.zeros((self.num_agents), dtype=int)
        self.num_defections = np.zeros((self.num_agents), dtype=int)
        self.next_match = next(self.gen)
        self.curr_round = 0
        print("Env Reset")
        return self._obs()

    def step(self, action_dict):
        done = False
        self.curr_round += 1
        match_players = self.next_match
        print(action_dict)
        # print("test print")
        a1_act = action_dict[match_players[0]]
        a2_act = action_dict[match_players[1]]
        self.num_match += 1
        self.last_acts[match_players[0]][match_players[1]] = a1_act
        self.last_acts[match_players[1]][match_players[0]] = a2_act
        self.turns_since_defect[match_players[0]] = (0 if a1_act == ACTIONS.DEFECT else self.turns_since_defect[match_players[0]] + 1)
        self.turns_since_defect[match_players[1]] = (0 if a2_act == ACTIONS.DEFECT else self.turns_since_defect[match_players[1]] + 1)
        self.num_defections[match_players[0]] += (1 if a1_act == ACTIONS.DEFECT else 0)
        self.num_defections[match_players[1]] += (1 if a2_act == ACTIONS.DEFECT else 0)
        self.agents_num_rounds[match_players[0]] += 1
        self.agents_num_rounds[match_players[1]] += 1

        if self.num_match == self.num_matches:
            if self.curr_round < self.rounds:
                self.num_match = 0
                self.gen = combinations(range(self.num_agents),2)
            else:
                print(f"Defect Ratio: {0 if self.agents_num_rounds[match_players[0]] == 0 else self.num_defections[match_players[0]] / self.agents_num_rounds[match_players[0]]}")
                done = True
        else:
            self.next_match = next(self.gen)
        rewards = {match_players[0] : self.get_reward(a1_act,a2_act), match_players[1]: self.get_reward(a2_act,a1_act)}
        self.agent_scores[match_players[0]] += rewards[match_players[0]]
        self.agent_scores[match_players[1]] += rewards[match_players[1]]
        self.total_score += rewards[match_players[0]] + rewards[match_players[1]]
        obs = self._obs() if not done else {}
        dones = {"__all__": done}
        infos = {}
        # print("Finished Step", obs, rewards, dones, infos)
        return obs, rewards, dones, infos

    def _obs(self):
        agent_id_0 = self.next_match[0]
        agent_id_1 = self.next_match[1]
        return_val = {
            agent_id_0 : self.agent_obs(agent_id_0,agent_id_1),
            agent_id_1 : self.agent_obs(agent_id_1,agent_id_0),
        }
        # print(return_val)
        return return_val

    def agent_obs(self, agent_id, opp_id):
        self.state[STATE_VARS.OPP_LAST_ACT] = self.last_acts[opp_id][agent_id]
        self.state[STATE_VARS.OPP_ID] = opp_id
        self.state[STATE_VARS.OPP_TURNS_SINCE_DEF] = self.turns_since_defect[opp_id]
        self.state[STATE_VARS.NM_AGENTS] = self.num_agents
        self.state[STATE_VARS.ENV_TOTAL_SCORE] = self.total_score
        self.state[STATE_VARS.CURR_TOTAL_SCORE] = self.agent_scores[agent_id]
        self.state[STATE_VARS.OPP_TOTAL_SCORE] = self.agent_scores[opp_id]
        self.state[STATE_VARS.OPP_DEF_RATE] = 0 if self.agents_num_rounds[opp_id] == 0 else self.num_defections[opp_id] / self.agents_num_rounds[opp_id]
        self.state[STATE_VARS.CURR_NUM_ROUNDS] = self.agents_num_rounds[agent_id]
        self.state[STATE_VARS.OPP_NUM_ROUNDS] = self.agents_num_rounds[opp_id]
        #self.state[STATE_VARS.AVG_AGENT_DEF_RATE] = np.average([self.num_defections[i] / self.agents_num_rounds[i] for i in self.num_agents])

        return np.copy(self.state)
        
    def get_reward(self, agent_act, opponent_act):
        # There is a simpler way to implement this if defect and cooperate are 0 and 1,
        # But this method should stay accurate if we decide to change how we represent defect and cooperate
        ind_reward = 0
        avg_total_reward = self.total_score / (self.curr_round * self.rounds + self.num_match)
        if agent_act == ACTIONS.DEFECT:
            if opponent_act == ACTIONS.COOPERATE:
                ind_reward = 4
            else:
                ind_reward = 1
        else:
            if opponent_act == ACTIONS.COOPERATE:
                ind_reward = 3
            else:
                ind_reward = 0
        return ind_reward + avg_total_reward


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