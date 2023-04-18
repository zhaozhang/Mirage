import logging
import math
from abc import abstractmethod
from sim.cluster import Cluster
from datetime import timedelta


class Policy:

    @abstractmethod
    def reconfig(self, job_logs, curr_time):
        pass

    @abstractmethod
    def update_complete_jobs(self, complete_jobs, curr_time):
        pass

    @abstractmethod
    def gen_submitted_jobs(self, pending_queue, avail_nodes, curr_time):
        pass


class BackfillPolicy(Policy):
    """
    Backfill parameters
    1. bf_interval=# - Interval between backfill scheduling attempts. Default value is 30 seconds
    2. bf_resolution=# - Time resolution of backfill scheduling. Default value is 60 seconds
    3. bf_window=# - How long into the future to look when determining when and where jobs can start. Default value is one day (minutes)
    4. bf_continue - If set, then continue backfill scheduling after periodically releasing locks for other operations
    5. bf_max_job_test=# - Maximum number of jobs consider for backfill scheduling. Default value is 100 jobs
    6. bf_max_job_start=# - Maximum number of jobs backfill schedule. Default value is 0 (no limit)
    7. bf_max_job_part=# - Maximum number of jobs per partition to consider for backfill scheduling. Default value is 0 (no limit)
    8. bf_max_job_user=# - Maximum number of jobs per user to consider for backfill scheduling. Default value is 0 (no limit)
    9. bf_yield_interval=# - Time between backfill scheduler lock release. Default value 2000000 usec (2 seconds, new option in version 14.11)
    10. bf_yield_sleep=# - Time that backfill scheduler sleeps for when locks are released. Default value 500000 usec (0.5 seconds, new option in version 14.11)
    11. max_rpc_cnt=# - Sleep if the number of pending RPCs reaches this level. Default value is 0 (no limit)

    Settings on longhorn/frontera
    1. bf_continue = 1
    2. bf_max_job_test = 3000
    3. bf_window = 2880
    4. bf_resolution = 600
    5. bf_max_time = 600
    6. bf_job_part_count_reserve = 100
    """

    def __init__(self, config, start_time, nodes):
        self._config = config
        self._time = start_time
        self._running_jobs = {}
        # timeline will be used only for counting the time slots
        self._timeline = [nodes for _ in range(0, int(config["bf_window"] * 60 / config["bf_resolution"]))]
        self._nodes = nodes
        self._time_count = 0

    def reconfig(self, job_logs, curr_time) -> None:
        self._running_jobs = {}
        self._time = curr_time
        time_period = int(self._config["bf_window"] * 60 / self._config["bf_resolution"])
        self._timeline = [self._nodes for _ in range(0, time_period)]
        for job_log in job_logs:
            self._running_jobs[job_log.job.job_id] = (job_log.original_end, job_log.job.nodes)
            for i in range(0, len(self._timeline)):
                if self._time + timedelta(seconds=i * self._config["bf_resolution"]) < job_log.original_end:
                    self._timeline[i] -= job_log.job.nodes

    def update_complete_jobs(self, complete_jobs, curr_time):
        # TODO: Update the virtual cluster state for jobs completed (From scheduler side)
        # When the cluster completes jobs, this virtual cluster state should be update
        """
        time_shift = int((curr_time - self._time).total_seconds() / self._config["bf_resolution"])
        for i in range(0, time_shift):
            self._timeline.pop(i)
            self._timeline.append(self._nodes)

        for key, value in self._running_jobs.items():
            limit_end_time = value[0]
            for i in range(0, time_shift):
                timeline_slot_start = self._time + timedelta(seconds=len(self._timeline) + i) * self._config[
                    "bf_resolution"]
                if timeline_slot_start < limit_end_time:
                    self._timeline[-(time_shift - i)] -= value[1]
        self._time += timedelta(seconds=time_shift * self._config["bf_resolution"])
        """
        for complete_job in complete_jobs:
            self._running_jobs.pop(complete_job.job.job_id)
            """
            job_completion_timeline = math.ceil(
                (complete_job.original_end - self._time).total_seconds() /
                self._config["bf_resolution"])
            if job_completion_timeline > 0:
                for i in range(0, job_completion_timeline):
                    self._timeline[i] += complete_job.job.nodes
            """
        return True

    def gen_submitted_jobs(self, pending_queue, avail_nodes, curr_time) -> []:
        # TODO: Determine which jobs will be submitted to the cluster
        submitted_jobs = []
        debug_time = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
        if self._time_count == 0:
            self._time = curr_time
            virtual_timeline = [self._nodes for _ in range(0, len(self._timeline))]
            for key, value in self._running_jobs.items():
                for i in range(0, len(self._timeline)):
                    if self._time + timedelta(seconds=i * self._config["bf_resolution"]) < value[0]:
                        virtual_timeline[i] -= value[1]
            backfill_max_job = len(pending_queue) if len(pending_queue) < self._config["bf_max_job_test"] else \
                self._config["bf_max_job_test"]
            for i in range(0, backfill_max_job):
                end_index = 0
                curr_index = 0
                is_scheduled = True
                while True:
                    if virtual_timeline[curr_index] >= pending_queue[i].nodes:
                        start_index = curr_index
                        longest_index = math.ceil(pending_queue[i].time_limit * 60 / self._config["bf_resolution"])
                        if longest_index + curr_index > len(virtual_timeline):
                            longest_index = len(virtual_timeline) - curr_index
                        for slot in range(0, longest_index):
                            if virtual_timeline[curr_index + slot] < pending_queue[i].nodes:
                                curr_index = curr_index + slot + 1
                                is_scheduled = False
                                break
                            else:
                                end_index = curr_index + slot
                        if is_scheduled:
                            for index in range(start_index, end_index + 1):
                                virtual_timeline[index] -= pending_queue[i].nodes
                            if start_index == 0:
                                if i != 0:
                                    logging.debug("Time: {}, Backfilling Submitted Jobs: {}".format(curr_time,
                                                                                                    pending_queue[
                                                                                                        i].job_id))
                                    pending_queue[i].backfill_time = curr_time
                                self._running_jobs[pending_queue[i].job_id] = (
                                    curr_time + timedelta(minutes=pending_queue[i].time_limit), pending_queue[i].nodes)
                                submitted_jobs.append(pending_queue[i])
                            break
                        else:
                            if curr_index >= len(virtual_timeline):
                                break
                            else:
                                is_scheduled = True
                                continue
                    else:
                        curr_index += 1
                        if curr_index >= len(virtual_timeline):
                            break
        else:
            curr_nodes = avail_nodes
            for i in range(0, len(pending_queue)):
                if curr_nodes >= pending_queue[i].nodes:
                    # end_slot = math.ceil(pending_queue[i].time_limit * 60 / self._config["bf_resolution"])
                    # debug_time = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
                    # for time_slot in range(0, min(end_slot, len(self._timeline))):
                    #    self._timeline[time_slot] -= pending_queue[i].nodes
                    #    debug_slot = time_slot
                    submitted_jobs.append(pending_queue[i])
                    self._running_jobs[pending_queue[i].job_id] = (
                        curr_time + timedelta(minutes=pending_queue[i].time_limit), pending_queue[i].nodes)
                    curr_nodes -= pending_queue[i].nodes
                else:
                    break
        self._time_count += 1
        if self._time_count == self._config["bf_interval"]:
            self._time_count = 0
        return submitted_jobs


class Scheduler:
    def __init__(self, nodes, init_time, policy, config, mode, queue_time_len=100):
        self._cluster = Cluster(nodes, init_time, mode)
        self._pending_queue = []
        self._policy = policy
        self._job_logs = []
        self._time = init_time
        self._config = config
        self._reward_pool = {}
        self._dependency_dict = {}
        self._queue_time = []
        self._mode = mode
        self._queue_time_len = queue_time_len

    @property
    def job_logs(self):
        return self._job_logs

    def reconfig(self, running_jobs_dict, pending_jobs_dict, start_time):
        self._time = start_time
        self._pending_queue = []
        for job_log in pending_jobs_dict.values():
            self._pending_queue.append(job_log.job)
        self._policy.reconfig(list(running_jobs_dict.values()), start_time)
        self._cluster.reconfig(running_jobs_dict, start_time)

    def submit(self, jobs):
        def priority_reset(new_job):
            """
            Initialize the job priority

            :param new_job: Job that need to be initialized
            :return: None
            """
            return new_job.priority - self._config["PriorityWeightAge"] * (
                (new_job.wait_time.total_seconds() / (float(self._config["PriorityMaxAge"]) * 24 * 60 * 60)))

        for job in jobs:
            if job.priority_from_log:
                job.priority = job.early_priority + priority_reset(job)
            else:
                job.priority = job.early_priority + self._config["PriorityWeightJobSize"] * (
                        job.nodes / self._cluster.num_proc)
            self._pending_queue.append(job)
        if len(jobs) != 0:
            self._pending_queue = sorted(self._pending_queue, key=lambda x: x.priority, reverse=True)
        return True

    def register_dependency(self, jobs):
        for job in jobs:
            if job.prev_job is not None:
                self._dependency_dict[job.prev_job] = job

    def run(self, time, job_id=None):
        # TODO: Run the scheduler and cluster for certain time
        job_end = False
        job_return = None
        for _ in range(0, int(time.total_seconds() / int(self._mode))):
            new_submit_jobs = self._policy.gen_submitted_jobs(self._pending_queue, self._cluster.available_proc,
                                                              self._time)
            for new_job in new_submit_jobs:
                self._pending_queue.remove(new_job)
                if new_job.prev_job is not None and new_job.prev_job_end is not None:
                    interruption = int((self._time - new_job.prev_job_end).total_seconds())
                    self._reward_pool[new_job.job_id] = (new_job.prev_job, interruption, self._time)
                    logging.info(
                        "Simulator time (Successor start time): {}, "
                        "Reward Pool Update: Predecessor end time: {}, "
                        "Interruption-Overlap (seconds): {}".format(self._time, new_job.prev_job_end, interruption))
                if len(self._queue_time) == self._queue_time_len:
                    self._queue_time.pop(0)
                self._queue_time.append((self._time - new_job.submit).total_seconds())
                if new_job.next_job is not None:
                    self._dependency_dict[new_job.job_id].prev_job_end = self._time + new_job.duration
            for new_job in new_submit_jobs:
                self._cluster.submit_job(new_job)
            self._time += timedelta(seconds=int(self._mode))
            for item in self._pending_queue:
                item.priority = item.priority + (self._config["PriorityWeightAge"] * int(self._mode) / (
                        float(self._config["PriorityMaxAge"]) * 24 * 60 * 60))
            run_logs = self._cluster.run_time(timedelta(seconds=int(self._mode)))
            self._policy.update_complete_jobs(run_logs, self._time)
            self._job_logs += run_logs

            if job_id is not None:
                for new_submit_job in new_submit_jobs:
                    if new_submit_job.job_id == job_id:
                        job_end = True
                        job_return = (job_id, self._time - timedelta(seconds=int(self._mode)))
                        break
        return job_end, job_return

    @property
    def cluster_state(self):
        return self._cluster.proc_status

    @property
    def running_jobs_state(self):
        return self._cluster.job_info

    @property
    def queue_state(self):
        return self._pending_queue

    @property
    def queue_size(self):
        return len(self._pending_queue)

    @property
    def reward_pool(self) -> {}:
        return self._reward_pool

    def avg_queue_time(self):
        return sum(self._queue_time) / len(self._queue_time)

    @property
    def pending_queue(self):
        return self._pending_queue

    @property
    def avail_nodes(self):
        return self._cluster.available_proc
