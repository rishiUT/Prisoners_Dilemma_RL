from enum import Enum, IntEnum, auto

class STATE_VARS(IntEnum):
    OPP_ID = 0
    OPP_DEF_RATE = auto()
    OPP_TURNS_SINCE_DEF = auto()
    OPP_LAST_ACT = auto()
    CURR_TOTAL_SCORE = auto()
    OPP_TOTAL_SCORE = auto()
    ENV_TOTAL_SCORE = auto()
    NM_AGENTS = auto()
    AVG_AGENT_DEF_RATE = auto()
    
class ACTIONS(IntEnum):
    COOPERATE = 0
    DEFECT = 1