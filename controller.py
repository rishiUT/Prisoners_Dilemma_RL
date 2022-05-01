import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from env import Environment, PDGame
from agent import *
import argparse
from ray import tune
import random
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

env_name = "PDGame"
register_env(env_name,lambda env_config: PDGame)
policies = {"cynic": PolicySpec(policy_class=CynicPolicy),
            #"beat_last": (BeatLastHeuristic, Discrete(3), Discrete(3), {}),
            "easy_mark": PolicySpec(policy_class=EasyMarkPolicy),
            "sac": PolicySpec(policy_class=SACTorchPolicy),
            "ppo": PolicySpec(policy_class=PPOTorchPolicy)
            }
def select_policy(agent_id,episode,worker,**kwargs):
    if agent_id == 0:
        return "ppo"

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
        "--stop-iters", type=int, default=3, help="Number of iterations to train."
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
                     "disable_env_checking": True,
                     "ignore_worker_failures": False,
                     "num_workers": 1,
                     "framework": "torch",
                     "num_envs_per_worker": 1,
                     "train_batch_size": 1,
                     "sgd_minibatch_size": 1,
                     #"multiagent": {"policies_to_train": ["learned"],
                     "multiagent": {"policies_to_train": ["ppo","sac"],
                                    "policies": policies,
                                    "policy_mapping_fn": select_policy,
                                   }
                    }
    ppo_trainer = PPOTrainer(
        env=PDGame,
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": select_policy,
                "policies_to_train": ["ppo"],
            },
            "model": {
                "vf_share_layers": True,
            },
            "num_sgd_iter": 6,
            "vf_loss_coeff": 0.01,
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "MeanStdFilter",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        },
    )

    sac_trainer = SACTrainer(
        env=PDGame,
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": select_policy,
                "policies_to_train": ["sac"],
            },
            "model": {
                "vf_share_layers": True,
            },
            "gamma": 0.95,
            # "n_step": 3,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        },
    )
    print("RUNNING")
    # tune.run("PPO", config=config,stop={"training_iteration": 10})
    for i in range(args.stop_iters):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- SAC --")
        result_sac = sac_trainer.train()
        print(result_sac)

        # improve the PPO policy
        print("-- PPO --")
        result_ppo = ppo_trainer.train()
        print(result_ppo)

        # Test passed gracefully.
        if (
            args.as_test
            and result_sac["episode_reward_mean"] > args.stop_reward
            and result_ppo["episode_reward_mean"] > args.stop_reward
        ):
            print("test passed (both agents above requested reward)")
            quit(0)

        # swap weights to synchronize
        sac_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        ppo_trainer.set_weights(sac_trainer.get_weights(["dqn_policy"]))
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