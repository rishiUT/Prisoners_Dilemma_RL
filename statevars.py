from enum import Enum, IntEnum, auto

class STATE_VARS(IntEnum):
    OPP_ID = auto()
    OPP_DEF_RATE = auto()
    OPP_TURNS_SINCE_DEF = auto()
    OPP_LAST_ACT = auto()
    CURR_TOTAL_SCORE = auto()
    ENV_TOTAL_SCORE = auto()
    NM_AGENTS = auto()
    ALL_AGENT_DEF_RATE = auto()
    
class ACTIONS(IntEnum):
    COOPERATE = 0
    DEFECT = 1