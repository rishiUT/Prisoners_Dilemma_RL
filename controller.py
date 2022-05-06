import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from env import  PDGame
from agent import *
from metrics import MyCallbacks
import argparse
from ray import tune
import ray

import random
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy

env_name = "PDGame"
register_env(env_name,lambda env_config: PDGame)
policies = {"cynic": PolicySpec(policy_class=CynicPolicy),
            #"beat_last": (BeatLastHeuristic, Discrete(3), Discrete(3), {}),
            "easy_mark": PolicySpec(policy_class=EasyMarkPolicy),
            "tft": PolicySpec(policy_class=TitForTatPolicy),
            "sac": PolicySpec(policy_class=SACTorchPolicy),
            "ppo": PolicySpec(policy_class=PPOTorchPolicy),
            "dqn": PolicySpec(policy_class=DQNTorchPolicy),
            }
def select_policy(agent_id,episode,worker,**kwargs):
    if agent_id == 0:
        return "ppo"

    elif agent_id == 1:
        return "tft"

    elif agent_id == 2:
        return "tft"

    elif agent_id == 3:
        return "tft"
        
    else:
        return random.choice(["cynic", "cynic"])
if __name__ == "__main__":
    
    from csv import writer
    to_print = ["EPISODES", "PPO", "TFT"]
    with open('test.csv', 'w', newline='') as f_object:  
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(to_print)  
            # Close the file object
            f_object.close()

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
        "--stop-iters", type=int, default=30, help="Number of iterations to train."
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

    ray.init(log_to_driver=True)
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
            "callbacks": MyCallbacks,
            "vf_loss_coeff": 0.01,
            # "train_batch_size": 1,
            
            # "rollout_fragment_length": 1,
            # "sgd_minibatch_size": 1,
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "MeanStdFilter",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        },
    )
    #Path to checkpoint if want to start from there 
    # ppo_trainer.restore("./ppo_ck/checkpoint_000020/checkpoint-20")

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
            "callbacks": MyCallbacks,
            # "train_batch_size": 1,
            # "sgd_minibatch_size": 1,
            # "n_step": 3,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        },
    )

    dqn_trainer = DQNTrainer(
        env=PDGame,
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": select_policy,
                "policies_to_train": ["dqn"],
            },
            "model": {
                "vf_share_layers": True,
            },
            "gamma": 0.95,
            "n_step": 3,
            # "train_batch_size": 1,
            # "rollout_fragment_length": 1,
            # "sgd_minibatch_size": 1,
            "callbacks": MyCallbacks,
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
        # print("-- SAC --")
        # result_sac = sac_trainer.train()
        # result_dqn = dqn_trainer.train()
        # print(result_dqn)

        # improve the PPO policy
        # print("-- PPO --")
        result_ppo = ppo_trainer.train()
        # print(result_ppo)

        # Test passed gracefully.
        if (
            args.as_test
            # and result_sac["episode_reward_mean"] > args.stop_reward
            # and result_dqn["episode_reward_mean"] > args.stop_reward
            and result_ppo["episode_reward_mean"] > args.stop_reward
        ):
            print("test passed (both agents above requested reward)")
            quit(0)

        # swap weights to synchronize
        # sac_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        # ppo_trainer.set_weights(sac_trainer.get_weights(["sac_policy"]))
        # ppo_trainer.set_weights(dqn_trainer.get_weights(result_dqn["dqn"]))
        # dqn_trainer.set_weights(ppo_trainer.get_weights(result_ppo["ppo_policy"]))
        # print(f"Per episode sac reward: {result_sac['episode_reward_mean']}")
        print(f"Per episode ppo reward: {result_ppo['episode_reward_mean']}")
        # print(f"Per episode dqn reward: {result_dqn['episode_reward_mean']}")
    ppo_trainer.save("ppo_ck")
    #dqn_trainer.save("dqn_ck")
    print("FINISHED")

