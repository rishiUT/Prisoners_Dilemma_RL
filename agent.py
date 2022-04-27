from statevars import STATE_VARS, ACTIONS
from ray.rllib.policy.policy import Policy


class CynicPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        print("Cynic Step")
        return [ACTIONS.DEFECT for _ in obs_batch], [], {}
    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass
class EasyMarkPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [ACTIONS.COOPERATE for _ in obs_batch], [], {}
    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats
    def get_weights(self):
        pass
    def set_weights(self, weights):
            pass
class TitForTat(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [ACTIONS.COOPERATE for _ in obs_batch], [], {}
    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats
    def get_weights(self):
        pass
    def set_weights(self, weights):
            pass
class Agent():
    env = None
    def __init__(self, agent_id) -> None:
        self.reward_total = 0
        self.id = agent_id
    
    def act(self, state) -> int:
        raise NotImplementedError()
    
    def get_reward(self, reward) -> None:
        raise NotImplementedError()

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

class Cynic(Agent):
    def __init__(self, state_size, agent_id ) -> None:
        super().__init__(agent_id)
        self.ss = state_size
    
    def act(self, state) -> int:
        print("The CYNIC defects.")
        return ACTIONS.DEFECT
    
    def get_reward(self, reward) -> None:
        self.reward_total += reward
        print("The current reward for the CYNIC is ", self.reward_total)

class EasyMark(Agent):
    def __init__(self, state_size, agent_id) -> None:
        super().__init__(agent_id)
        self.ss = state_size
    
    def act(self, state) -> int:
        print("The EASY MARK cooperates.")
        return ACTIONS.COOPERATE
    
    def get_reward(self, reward) -> None:
        self.reward_total += reward
        print("The current reward for the EASY MARK is ", self.reward_total)

class TitForTat(Agent):
    def __init__(self, state_size, agent_id) -> None:
        super().__init__(agent_id)
        self.ss = state_size
    
    def act(self, state) -> int:
        action = ACTIONS.DEFECT if int(state[STATE_VARS.OPP_LAST_ACT]) == ACTIONS.DEFECT else ACTIONS.COOPERATE
        print(f"The TFT {'defects' if action is ACTIONS.DEFECT else 'cooperates'}")
        return action

    
    def get_reward(self, reward) -> None:
        self.reward_total += reward
        print("The current reward for the TFT is ", self.reward_total)
