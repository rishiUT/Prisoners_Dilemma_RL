import agent
from agent import Cynic
from agent import Easy_Mark
import numpy as np

state_size = 0
state = np.array(state_size)

class Environment():
    def __init__(self) -> None:
        self.agents = []
        self.agents.append(Cynic(1))
        self.agents.append(Easy_Mark(1))
        self.total_reward = 0
    
    def run_sim(self) -> None:
        for agent in self.agents:
            for opponent in self.agents:
                if opponent is not agent:
                    agent_action = agent.act(get_state(agent, opponent))
                    opp_action = opponent.act(get_state(opponent, agent))
                    agent_reward = get_reward(agent_action, opp_action)
                    opp_reward = get_reward(opp_action, agent_action)
                    agent.get_reward(agent_reward)
                    opponent.get_reward(opp_reward)
                    self.total_reward += agent_reward + opp_reward


def get_state(agent, opponent):
    #This is incomplete, once we have state variables we can determine how they're determined
    return state

def get_reward(agent_act, opponent_act):
    # There is a simpler way to implement this if defect and cooperate are 0 and 1,
    # But this method should stay accurate if we decide to change how we represent defect and cooperate
    if agent_act == agent.DEFECT:
        if opponent_act == agent.COOPERATE:
            return 3
        else:
            return 1
    else:
        if opponent_act == agent.COOPERATE:
            return 2
        else:
            return 0