
from typing import Dict, Tuple
import argparse
import numpy as np
import os
from statistics import mean
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=2000)


class MyCallbacks(DefaultCallbacks):
    # def on_episode_start(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs
    # ):
    #     # Make sure this episode has just been started (only initial obs
    #     # logged so far).
    #     assert episode.length == 0, (
    #         "ERROR: `on_episode_start()` callback should be called right "
    #         "after env reset!"
    #     )
    #     # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
    #     episode.user_data["action_dist"] = []
    #     episode.hist_data["action_dist"] = []

    # def on_episode_step(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs
    # ):
    #     # Make sure this episode is ongoing.
    #     assert episode.length > 0, (
    #         "ERROR: `on_episode_step()` callback should not be called right "
    #         "after env reset!"
    #     )
    #     action = episode.last_action_for()
    #     # print(f"action {action}")
    #     episode.user_data["action_dist"].append(action)
    #     episode.hist_data["action_dist"].append(action)
        

    # def on_episode_end(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs
    # ):
    #     # Check if there are multiple episodes in a batch, i.e.
    #     # "batch_mode": "truncate_episodes".
    #     if worker.policy_config["batch_mode"] == "truncate_episodes":
    #         # Make sure this episode is really done.
    #         assert episode.batch_builder.policy_collectors["default_policy"].batches[
    #             -1
    #         ]["dones"][-1], (
    #             "ERROR: `on_episode_end()` should only be called "
    #             "after episode is done!"
    #         )
    #     pole_angle = np.mean(episode.user_data["pole_angles"])
    #     print(
    #         "episode {} (env-idx={}) ended with length {} and pole "
    #         "angles {}".format(
    #             episode.episode_id, env_index, episode.length, pole_angle
    #         )
    #     )
    #     episode.custom_metrics["pole_angle"] = pole_angle
    #     episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print(
    #         "trainer.train() result: {} -> {} episodes".format(
    #             trainer, result["episodes_this_iter"]
    #         )
    #     )
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     print(
    #         "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #             policy, result["sum_actions_in_train_batch"]
    #         )
    #     )

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        # print("Batches ", original_batches[agent_id][1]['actions'])
        agent_actions = original_batches[agent_id][1]['actions']
        if f"agent_{agent_id}_num_actions" not in episode.custom_metrics:
            episode.custom_metrics[f"agent_{agent_id}_num_actions"] = 0
            episode.custom_metrics[f"agent_{agent_id}_num_defect"] = 0
        episode.custom_metrics[f"agent_{agent_id}_num_actions"] += len(agent_actions)
        episode.custom_metrics[f"agent_{agent_id}_num_defect"] += sum(agent_actions)
        episode.custom_metrics[f"agent_{agent_id}_defect_ratio"] = episode.custom_metrics[f"agent_{agent_id}_num_defect"] / episode.custom_metrics[f"agent_{agent_id}_num_actions"]
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
            episode.custom_metrics["num_batches"] += 1
        # print(f"Ratio: {episode.custom_metrics[f'agent_{agent_id}_defect_ratio']}")
        # episode.custom_metrics["defect_ratio"] = mean(episode.user_data["action_dist"])



if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()
    trials = tune.run(
        "PG",
        stop={
            "training_iteration": args.stop_iters,
        },
        config={
            "env": "CartPole-v0",
            "num_envs_per_worker": 2,
            "callbacks": MyCallbacks,
            "framework": args.framework,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        },
    ).trials

    # Verify episode-related custom metrics are there.
    custom_metrics = trials[0].last_result["custom_metrics"]
    print(custom_metrics)
    assert "pole_angle_mean" in custom_metrics
    assert "pole_angle_min" in custom_metrics
    assert "pole_angle_max" in custom_metrics
    assert "num_batches_mean" in custom_metrics
    assert "callback_ok" in trials[0].last_result

    # Verify `on_learn_on_batch` custom metrics are there (per policy).
    if args.framework == "torch":
        info_custom_metrics = custom_metrics["default_policy"]
        print(info_custom_metrics)
        assert "sum_actions_in_train_batch" in info_custom_metrics