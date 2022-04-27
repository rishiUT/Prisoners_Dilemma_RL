import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from env import Environment, PDGame
from agent import *
import argparse
from ray import tune
import random
from ray.tune.registry import register_env
env_name = "PDGame"
register_env(env_name,lambda env_config: PDGame)

def select_policy(agent_id):
    print(f"selecting agent {agent_id}")
    if agent_id == 0:
        return "cynic"

    elif agent_id == 1:
        return "cynic"

    else:
        return random.choice(["cynic", "cynic"])
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=150, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=1000.0,
        help="Reward at which we stop training.",
    )
    parser.add_argument('-agents', '--agents', nargs='+', default=[])
    args = parser.parse_args()
    config={"env": PDGame,
                     #"eager": True,
                     "gamma": 0.9,
                     "disable_env_checking": False,
                     "num_workers": 1,
                     "framework": "torch",
                     "num_envs_per_worker": 1,
                     "train_batch_size": 128,
                     #"multiagent": {"policies_to_train": ["learned"],
                     "multiagent": {"policies_to_train": [],
                                    "policies": {"cynic": (CynicPolicy, PDGame.observation_space, PDGame.action_space, {}),
                                                 #"beat_last": (BeatLastHeuristic, Discrete(3), Discrete(3), {}),
                                                 "easy_mark": (EasyMarkPolicy, PDGame.observation_space, PDGame.action_space, {}),
                                                 },
                                    "policy_mapping_fn": select_policy,
                                   }
                    }
    print("RUNNING")
    tune.run("PPO", config=config)
    print("FINISHED")


# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-agents', '--agents', nargs='+', default=[])
#     args = parser.parse_args()
#     agents = []
#     for idx,agent in enumerate(args.agents):
#         agents.append(globals()[args.a1](1,idx))
#         agents.append(globals()[args.a2](1,idx))
#     env = Environment(agents)
#     Agent.env = env
#     env.run_sim()