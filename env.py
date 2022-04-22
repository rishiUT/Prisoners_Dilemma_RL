import agent
from agent import *
import numpy as np
from statevars import STATE_VARS




class Environment():
    def __init__(self,agents) -> None:
        state_size = len(list(STATE_VARS))
        
        
        self.agents = agents
        if not self.agents:
            self.agents.append(Cynic(1,0))
            self.agents.append(EasyMark(1,1))
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