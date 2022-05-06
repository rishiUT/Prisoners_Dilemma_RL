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
        # print("Cynic Step")
        # # print(f"batch_size: {len(obs_batch)}")
        ret_action = [ACTIONS.DEFECT for _ in obs_batch]
        # print(f"Cynic Action: {ret_action[0]}")
        return ret_action, [], {}
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

class TitForTatPolicy(Policy):
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
        return [ACTIONS.COOPERATE if int(obs[STATE_VARS.OPP_LAST_ACT]) == int(ACTIONS.COOPERATE) else ACTIONS.DEFECT for obs in obs_batch ], [], {}
    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass
    