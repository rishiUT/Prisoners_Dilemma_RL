from argparse import Action
import os

from torch import randint
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import agent
from agent import *
import numpy as np
from statevars import STATE_VARS, ACTIONS, REWARD
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
        self.num_agents = 4
        self.reward_type = REWARD.TEAM
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
        self.last_act_overall = np.zeros((self.num_agents,),dtype=int)
        self.turns_since_defect = np.zeros((self.num_agents), dtype=int)
        self.agents_num_rounds = np.zeros((self.num_agents), dtype=int)
        self.total_score = 0
        self.agent_scores = np.zeros((self.num_agents), dtype=int)
        self.num_defections = np.zeros((self.num_agents), dtype=int)
        self.state = np.zeros((self.state_size))
        self.curr_round = 0
        self.num_resets = 0

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
        self.rounds = random.randint(2, 10)
        #print("Env Reset")
        self.num_resets += 1
        return self._obs()

    def print_to_csv(self, action_dict):
        to_print = [self.num_resets,]
        for i in range(self.num_agents):
            if i in action_dict.keys():
                to_print.append(action_dict[i])
            else:
                to_print.append('-')
        
        # to_print.append(self.last_acts[action_dict.keys()[1]][action_dict.keys()[0]])

        from csv import writer
        with open('test.csv', 'a', newline='') as f_object:  
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(to_print)  
            # Close the file object
            f_object.close()


    def step(self, action_dict):
        done = False
        self.curr_round += 1
        match_players = self.next_match
        #if 0 in action_dict.keys() and 1 in action_dict.keys():
            #print(action_dict)

        self.print_to_csv(action_dict)

        a1_act = action_dict[match_players[0]]
        a2_act = action_dict[match_players[1]]
        self.num_match += 1
        self.last_acts[match_players[0]][match_players[1]] = a1_act
        self.last_acts[match_players[1]][match_players[0]] = a2_act
        self.last_act_overall[match_players[0]] = a1_act
        self.last_act_overall[match_players[1]] = a2_act
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
                #print(f"Defect Ratio: {0 if self.agents_num_rounds[match_players[0]] == 0 else self.num_defections[match_players[0]] / self.agents_num_rounds[match_players[0]]}")
                done = True
        else:
            self.next_match = next(self.gen)
        # rewards = {match_players[0] : self.get_reward(a1_act,a2_act), match_players[1]: self.get_reward(a2_act,a1_act)}
        rewards = {match_players[0] : self.get_reward(a1_act,a2_act,match_players[0],match_players[1]), match_players[1]: self.get_reward(a2_act,a1_act,match_players[1],match_players[0])}
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
        self.state[STATE_VARS.SELF_TEAM] = agent_id % 2
        self.state[STATE_VARS.OPP_TEAM] = opp_id % 2
        self.state[STATE_VARS.OPP_LAST_ACT_OVERALL] = self.last_act_overall[opp_id]
        #self.state[STATE_VARS.AVG_AGENT_DEF_RATE] = np.average([self.num_defections[i] / self.agents_num_rounds[i] for i in self.num_agents])

        return np.copy(self.state)
        
    def get_reward(self, agent_act, opponent_act, agent_id, opp_id):
        # There is a simpler way to implement this if defect and cooperate are 0 and 1,
        # But this method should stay accurate if we decide to change how we represent defect and cooperate
        extra_reward = 0
        ind_reward = 0
        # avg_total_reward = 0 # self.total_score / (self.curr_round * self.rounds + self.num_match)
        
        if self.reward_type == REWARD.COMMUNISM:
            return self.get_default_reward(agent_act, opponent_act, agent_id, opp_id) + self.get_default_reward(opponent_act, agent_act, opp_id, agent_id)
        elif self.reward_type == REWARD.TEAM:
            return self.get_team_reward(agent_act, opponent_act, agent_id, opp_id)
        elif self.reward_type == REWARD.SOCIALISM:
            extra_reward = self.get_default_reward(agent_act, opponent_act, agent_id, opp_id) + self.get_default_reward(opponent_act, agent_act, opp_id, agent_id)

        # DEFAULT PD Reward
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
        return extra_reward * 0.1 + ind_reward #ind_reward + avg_total_reward
    def get_default_reward(self, agent_act, opponent_act, agent_id, opp_id):
        # There is a simpler way to implement this if defect and cooperate are 0 and 1,
        # But this method should stay accurate if we decide to change how we represent defect and cooperate

        # DEFAULT PD Reward
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
        return ind_reward #ind_reward + avg_total_reward

    def get_team_reward(self, agent_act, opponent_act, agent_id, opp_id):
        # There is a simpler way to implement this if defect and cooperate are 0 and 1,
        # But this method should stay accurate if we decide to change how we represent defect and cooperate
        t1 = agent_id % 2
        t2 = opp_id % 2
        ind_reward = 0
        team_reward =  100 if t1 == t2 else -100 
        # self.total_score / (self.curr_round * self.rounds + self.num_match)
        if agent_act == ACTIONS.DEFECT:
            team_reward *= -1
            if opponent_act == ACTIONS.COOPERATE:
                ind_reward = 4
            else:
                ind_reward = 1
        else:
            if opponent_act == ACTIONS.COOPERATE:
                ind_reward = 3
            else:
                ind_reward = 0
        return ind_reward + team_reward

