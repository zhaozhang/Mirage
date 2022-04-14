import argparse
import datetime
import logging

import ray
import pickle

from simulator import *
from parser_log import *

# Inspiration: https://github.com/honnibal/spacy-ray/pull/
# 1/files#diff-7ede881ddc3e8456b320afb958362b2aR12-R45
from asyncio import Event
from typing import Tuple
from time import sleep

import ray
# For typing purposes
from ray.actor import ActorHandle
from tqdm import tqdm


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


def pre_simulate(simulator_start_time, workload_start_time, workload_end_time, slurm_config, backfill_config):
    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    job_log = "./filtered-longhorn-v100.log"
    jobs = parse_job(job_log)

    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs

    slurm_simulator.init_scheduler(88, slurm_config, simulator_start_time, backfill_config)

    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)
    # ---------------------------------------------------------------------------------------------------------------

    time = workload_start_time + timedelta(days=2)
    job_1 = job.Job("1000000", 1, None, time, None, 2880, 6000, False)
    slurm_simulator.submit_job_external([job_1])
    slurm_simulator.run_end(job_1.job_id)
    job_end_time = slurm_simulator.sim_time + job_1.duration - timedelta(minutes=1)
    # slurm_simulator.run_time(timedelta(days=2))
    return time + timedelta(minutes=1), job_end_time, int(slurm_simulator.avg_queue_time() / 60)


def simulate_per_sample(sample_time, simulator_start_time, workload_start_time, workload_end_time, slurm_config,
                        backfill_config):
    def infer_wrapper(simulator_time):
        if simulator_time == sample_time:
            return True
        return False

    def train_wrapper(batch, cache):
        cache.append(batch)
        return

    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    job_log = "./filtered-longhorn-v100.log"
    jobs = parse_job(job_log)

    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs

    slurm_simulator.init_scheduler(88, slurm_config, simulator_start_time, backfill_config)

    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)
    # ---------------------------------------------------------------------------------------------------------------

    time = workload_start_time + timedelta(days=2)
    job_1 = job.Job("1000000", 1, None, time, None, 2880, 6000, False)
    job_2 = job.Job("1000001", 1, None, time, None, 2880, 6000, False)
    batch_cache = []

    slurm_simulator.run_group_dependency([[job_1, job_2]],
                                         infer_func=lambda: infer_wrapper(slurm_simulator.sim_time),
                                         train_func=lambda x: train_wrapper(x, batch_cache),
                                         sample_time_length=600,
                                         train_freq=10,
                                         infer_freq=1,
                                         max_store_reward=1,
                                         sample_time_step=144,
                                         infer_lower_bound=-600,
                                         infer_upper_bound=600,
                                         offline=True,
                                         offline_sample_index=sample_time)
    return batch_cache


@ray.remote
def simulate_per_thread(init_time, i, slurm_config, backfill_config, actor, num_probe, start_time_interval, mode=False):
    data = []
    simulator_start_time = init_time + timedelta(hours=i * start_time_interval)
    workload_start_time = simulator_start_time
    workload_end_time = workload_start_time + timedelta(days=5)
    lower_bound, upper_bound, avg_time = pre_simulate(simulator_start_time, workload_start_time, workload_end_time,
                                                      slurm_config,
                                                      backfill_config)
    if mode:
        sample_point = [upper_bound - timedelta(minutes=avg_time)]
    else:
        interval = (upper_bound - lower_bound).total_seconds() / (num_probe - 1.0)
        sample_point = [(lower_bound + timedelta(seconds=interval * j)).replace(second=0) for j in range(0, num_probe)]
    for point in sample_point:
        data.append(
            simulate_per_sample(point, simulator_start_time, workload_start_time, workload_end_time,
                                slurm_config, backfill_config)[0][0])
    actor.update.remote(1)
    return data, avg_time


def simulate(num_probe, start_time_interval, parallel, num_sample_all, mode=False):
    sample_data = []
    sample_avg = []
    slurm_config = "./slurm_config.json"
    backfill_config = "./backfill_config.json"
    slurm_config = load_slurm_config(slurm_config)
    backfill_config = load_backfill_config(backfill_config)
    simulator_init_time = datetime.strptime("2021-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    if not parallel:
        # Do not use non-parallel version for this script!
        for i in range(0, num_sample_all):
            simulator_start_time = simulator_init_time + timedelta(hours=i * start_time_interval)
            workload_start_time = simulator_start_time
            workload_end_time = workload_start_time + timedelta(days=3)
            lower_bound, upper_bound, _ = pre_simulate(simulator_start_time, workload_start_time, workload_end_time,
                                                       slurm_config, backfill_config)
            interval = (upper_bound - lower_bound).total_seconds() / (num_probe - 1.0)
            sample_point = [(lower_bound + timedelta(seconds=interval * j)).replace(second=0) for j in
                            range(0, num_probe)]
            for point in sample_point:
                sample_data.append(
                    simulate_per_sample(point, simulator_start_time, workload_start_time, workload_end_time,
                                        slurm_config, backfill_config)[0][0][0])
    else:
        ray.init(runtime_env={"py_modules": ["."]})
        num_ticks = num_sample_all
        pb = ProgressBar(num_ticks)
        actor = pb.actor
        rpc = []
        for i in range(0, num_sample_all):
            rpc.append(
                simulate_per_thread.remote(simulator_init_time, i, slurm_config, backfill_config, actor, num_probe,
                                           start_time_interval, mode))
        pb.print_until_done()
        sample_data_tmp = ray.get(rpc)
        for item in sample_data_tmp:
            for s in item[0]:
                sample_data.append(s)
            sample_avg.append(item[1])
    return sample_data, sample_avg


def main():
    curr_time = datetime.today()
    parser = argparse.ArgumentParser()
    parser.add_argument("-parallel", action="store_true", default=False)
    parser.add_argument("-num_samples", type=int, default=2)
    parser.add_argument("-num_probe", type=int, default=5)
    parser.add_argument("-interval", "--start_time_interval", type=int, default=6)
    parser.add_argument("-baseline", "--avg_baseline", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    data, avg = simulate(args.num_probe, args.start_time_interval, args.parallel, args.num_samples, args.avg_baseline)
    # data, avg_queue_time = simulate_avg(args.start_time_interval, args.num_samples, args.parallel, args.num_probe)

    with open("batch_2800_7.pickle", "wb") as f:
        pickle.dump(data, f)
    with open("avg.json", "w") as f:
        json.dump(avg, f)

    print(datetime.today() - curr_time)


if __name__ == '__main__':
    main()
