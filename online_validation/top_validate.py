import pickle
import sys
import numpy as np
import torch
import ray
import argparse
import os
import logging

from simulator import *
from parser_log import *
from predictor import Predictor
from predictor import raw_features_to_np_ndarray

# Inspiration: https://github.com/honnibal/spacy-ray/pull/
# 1/files#diff-7ede881ddc3e8456b320afb958362b2aR12-R45
from asyncio import Event
from typing import Tuple
from time import sleep

# For typing purposes
from ray.actor import ActorHandle
from tqdm import tqdm

LINEAR_NEURAL_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,

    'n_hidden_units': [512, 256, 128, 64],
    'activation': torch.nn.LeakyReLU,
}

TRANSFORMER_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,

    'embed_size': 512,
    'encoder_nlayers': 2,
    'encoder_nheads': 2,
    'dim_feedforward': 128,
}

CONVOLUTION_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,
}


############################################################
# This is the Ray "actor" that can be called from anywhere to update
# our progress. You'll be using the `update` method. Don't
# instantiate this class yourself. Instead,
# it's something that you'll get from a `ProgressBar`.


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


######################################################################
# This is where the progress bar starts. You create one of these
# on the head node, passing in the expected total number of items,
# and an optional string description.
# Pass along the `actor` reference to any remote task,
# and if they complete ten
# tasks, they'll call `actor.update.remote(10)`.

# Back on the local node, once you launch your remote Ray tasks, call
# `print_until_done`, which will feed everything back into a `tqdm` counter.


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


@ray.remote
def simulate(args, actor, loop, infer_wrapper):
    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    job_log = "./filtered-frontera-rtx.log"
    slurm_config = "./slurm_config.json"
    backfill_config = "./backfill_config.json"
    slurm_config = load_slurm_config(slurm_config)
    backfill_config = load_backfill_config(backfill_config)

    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.load_jobs(job_log)
    # ---------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------------
    # Inference procedure
    start_time = datetime.strptime("2021-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S") + timedelta(
        hours=loop * args.interval)
    logging.info(f"Loop: {loop}, Start Time: {start_time}")
    workload_start_time = start_time
    workload_end_time = workload_start_time + timedelta(days=5)
    slurm_simulator.reset_history()
    slurm_simulator.reset_submitted_jobs()
    slurm_simulator.init_scheduler(88, slurm_config, start_time, backfill_config)
    workload = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(workload)

    time = workload_start_time + timedelta(days=2)
    job_1 = job.Job("1000000", 1, None, time, None, 2880, 6000, False)
    job_2 = job.Job("1000001", 1, None, time, None, 2880, 6000, False)

    try:
        slurm_simulator.run_group_dependency([[job_1, job_2]],
                                             infer_func=infer_wrapper,
                                             train_func=None,
                                             sample_time_length=600,
                                             infer_freq=10,
                                             sample_time_step=144,
                                             infer_lower_bound=-0.5)
    except:
        actor.update.remote(1)
        return None, None
    actor.update.remote(1)
    return slurm_simulator.reward(), slurm_simulator.prediction_trace


def main():
    # ---------------------------------------------------------------------------------------------------------------
    # Argument Parse
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_validate", type=int, default=1)
    parser.add_argument("-interval", type=int, default=4)
    args = parser.parse_args()

    # ---------------------------------------------------------------------------------------------------------------
    # Agent Initialization
    checkpoint_path = "../model_2800_7_convolution.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model_name = checkpoint['model_name']
    model_hparams = checkpoint['model_hparams']
    model_state = checkpoint['model_state_dict']
    optimizer_name = checkpoint['optimizer_name']
    optimizer_state = checkpoint['optimizer_state_dict']

    predictor = Predictor(model_name=model_name, hparams=model_hparams, optimizer_name=optimizer_name,
                          model_state=model_state, optimizer_state=optimizer_state,
                          base_learning_rate=1e-4, device='gpu')

    def infer_wrapper(data_input):
        np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False)

        # From numpy array to pytorch tensor
        tensor_input = torch.from_numpy(np_ndarray_input)
        # logging.debug(f"Complete inference for one job pair")
        return predictor.inference(tensor_input).data[0].item()

    ray.init(runtime_env={"py_modules": ["."]})
    num_ticks = args.num_validate
    pb = ProgressBar(num_ticks)
    actor = pb.actor
    rpc = []
    rpc_counter = 0
    for i in range(0, args.num_validate):
        rpc.append(
            simulate.remote(args, actor, i, infer_wrapper))
        rpc_counter += 1
        print(rpc_counter)
    print("Start...")
    pb.print_until_done()
    sample_data_tmp = ray.get(rpc)
    prediction_trace_data = []
    total_reward = []
    for item in sample_data_tmp:
        total_reward += item[0]
        prediction_trace_data.append(item[1])
    reward_pool = total_reward
    with open("reward.pickle", "wb") as f:
        pickle.dump(total_reward, f)
    with open("reward.log", "w") as f:
        f.write(
            "------------------------------------------Reward Logs-------------------------------------------------\n")
        f.write("pred_job_id,succ_job_id,expected_reward,actual_reward,reward_gen_time\n")
        for item in reward_pool:
            if item is None:
                f.write("None\n")
            else:
                f.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]}\n")
        f.write(
            "---------------------------------------------------------------------------------------------------")
    with open("prediction_trace.tickle", "wb") as f:
        pickle.dump(prediction_trace_data, f)


if __name__ == '__main__':
    main()
