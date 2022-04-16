COOPERATE = 0
DEFECT = 1

class Agent():
    def __init__(self, state_size) -> None:
        pass
    
    def act(self, state) -> int:
        raise NotImplementedError()
    
    def get_reward(self, reward) -> None:
        raise NotImplementedError()

class Cynic(Agent):
    def __init__(self, state_size) -> None:
        self.ss = state_size
    
    def act(self, state) -> int:
        return DEFECT
    
    def get_reward(self, reward) -> None:
        pass

class Easy_Mark(Agent):
    def __init__(self, state_size) -> None:
        self.ss = state_size
    
    def act(self, state) -> int:
        return COOPERATE
    
    def get_reward(self, reward) -> None:
        pass