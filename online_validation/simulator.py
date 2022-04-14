import queue
import scheduler
import parser_log
import logging
from datetime import timedelta
from job import JobStatus
from datetime import datetime
import copy
import enum


class Mode(enum.IntEnum):
    SECONDS = 1
    MINUTES = 60


class StateInput:
    def __init__(self, curr_time, job_log_lst):
        self.curr_time = curr_time
        self.running_jobs = {}
        self.pending_jobs = {}
        for new_job in job_log_lst:
            if new_job.status == JobStatus.RUNNING:
                self.running_jobs[new_job.job.job_id] = new_job
            elif new_job.status == JobStatus.PENDING:
                self.pending_jobs[new_job.job.job_id] = new_job
            else:
                logging.warning("Invalid jobs for the state reconfiguration")

    def num_jobs(self):
        return len(self.running_jobs) + len(self.pending_jobs)


class StateOutput:
    def __init__(self, curr_time, running_jobs, pending_jobs, cluster_state):
        self.time = curr_time
        self.running_jobs = running_jobs
        self.pending_jobs = pending_jobs
        self.cluster_state = cluster_state


class Simulator:
    def __init__(self, mode=Mode.SECONDS):
        self._jobs = {}
        self._queue = queue.PriorityQueue()
        self._num_submitted_jobs = 0
        self._scheduler = scheduler.Scheduler(0, None, None, None, mode)
        self._time = None
        self._start_time = None
        self._active_reward = {}
        self._history_cache = {}
        self._global_history = []
        self._mode = mode
        self._reward = []
        self._prediction_trace = []

    def load_jobs(self, log):
        self._jobs = parser_log.parse_job(log)

    def init_scheduler(self, nodes, slurm_config, start_time, backfill_config):
        backfill_policy = scheduler.BackfillPolicy(backfill_config, start_time, nodes)
        self._scheduler = scheduler.Scheduler(nodes, start_time, backfill_policy, slurm_config, self._mode)
        self._time = start_time
        self._start_time = start_time

    def run_end(self, end_job_id=None):
        while True:
            if self._queue.empty():
                while True:
                    debug_time = self._time.strftime("%Y-%m-%dT%H:%M:%S")
                    end, end_job_info = self._scheduler.run(timedelta(seconds=int(self._mode)), end_job_id)
                    self._time += timedelta(seconds=int(self._mode))
                    if end_job_id is not None:
                        if end:
                            return end_job_info[1]
                    if len(self._scheduler.job_logs) == self._num_submitted_jobs:
                        return None
            else:
                next_submit_time, next_job_id, next_new_job = self._queue.get()
                submit_job = [next_new_job]
                while True:
                    if self._queue.empty():
                        break
                    submit_time, job_id, new_job = self._queue.get()
                    if submit_time == next_submit_time:
                        submit_job.append(new_job)
                    else:
                        self._queue.put((submit_time, job_id, new_job))
                        break
                end, end_job_info = self._scheduler.run(next_submit_time - self._time, end_job_id)
                self._scheduler.submit(submit_job)
                if end:
                    self._time = end_job_info[1] + timedelta(seconds=int(self._mode))
                else:
                    self._time = next_submit_time
                if end_job_id is not None:
                    if end:
                        return end_job_info[1]

    def run_time(self, time):
        for _ in range(0, int(time.total_seconds() / int(self._mode))):
            debug_time = self._time.strftime("%Y-%m-%dT%H:%M:%S")
            submit_jobs = []
            while True:
                if self._queue.empty():
                    break
                submit_time, job_id, new_job = self._queue.get()
                if submit_time == self._time:
                    submit_jobs.append(new_job)
                elif submit_time < self._time:
                    logging.warning(
                        "Error simulation: Job ID: {}, Simulator time: {}, Job submission time: {}".format(job_id,
                                                                                                           self._time,
                                                                                                           submit_time))
                else:
                    self._queue.put((submit_time, job_id, new_job))
                    break
            self._scheduler.submit(submit_jobs)
            self._scheduler.run(timedelta(seconds=int(self._mode)))
            self._time += timedelta(seconds=int(self._mode))
        return True

    def submit_job_internal(self, job_lst) -> bool:
        success = True
        for job_id in job_lst:
            try:
                if self._mode == Mode.MINUTES:
                    self._jobs[job_id].submit = self._jobs[job_id].submit.replace(second=0)
                if self._jobs[job_id].submit < self._time:
                    logging.warning("Job submission time is earlier than the simulator time. Job: {}".format(job_id))
                    self._jobs[job_id].submit = self._time
                self._queue.put((self._jobs[job_id].submit, job_id, self._jobs[job_id]))
                self._num_submitted_jobs += 1
            except IndexError:
                logging.error("Job ID doesn't not exist: {}".format(job_id))
                success = False
        return success

    def submit_job_external(self, job_lst):
        for job in job_lst:
            if self._mode == Mode.MINUTES:
                job.submit = job.submit.replace(second=0)
            self._queue.put((job.submit, job.job_id, job))
            self._num_submitted_jobs += 1

    def set_scheduler_state(self, state_input):
        # Need to init scheduler first
        self._scheduler.reconfig(state_input.running_jobs, state_input.pending_jobs, state_input.curr_time)

    def reset_submitted_jobs(self):
        self._queue = queue.PriorityQueue()
        self._num_submitted_jobs = 0

    def reset_history(self):
        self._active_reward = {}
        self._history_cache = {}
        self._global_history = []

    def reconfig_exec(self, state_input, test_job, next_start_time=None, next_end_time=None) -> datetime:
        self.set_scheduler_state(state_input)
        self._num_submitted_jobs = state_input.num_jobs()
        self.submit_job_external([test_job])
        if next_start_time is not None and next_end_time is not None:
            self.find_jobs(next_start_time, next_end_time)
        elif next_start_time is None and next_end_time is None:
            pass
        else:
            logging.warning(
                "Exec_start_time method needs to have both start time and end time after the self-defined job is \
                submitted or these should be None")
        job_start_time_info = self.run_end(test_job.job_id)
        return job_start_time_info[1]

    def find_jobs(self, start_time, end_time) -> []:
        job_lst = []
        for key, value in self._jobs.items():
            if start_time <= value.submit <= end_time:
                job_lst.append(key)
        return job_lst

    def output_state(self) -> StateOutput:
        return StateOutput(self._time, self._scheduler.running_jobs_state, self._scheduler.queue_state,
                           self._scheduler.cluster_state)

    def run_job_dependency(self, first_job, second_job, next_submit_func, step_sec=1, timeout=None) -> (
            bool, datetime):
        first_job.submit = self._time
        self._scheduler.submit([first_job])
        if timeout is not None:
            end_time = self._time + timedelta(minutes=timeout)
        else:
            end_time = None
        while True:
            if end_time is not None:
                if self._time >= end_time:
                    return False, None
            self.run_time(timedelta(seconds=step_sec))
            if next_submit_func():
                second_job.submit = self._time
                self._scheduler.submit([second_job])
                # Run until the second job start to execute
                end_job_time = self.run_end(second_job.job_id)
                return True, end_job_time

    def run_group_dependency(self, groups, infer_func, train_freq=300, infer_freq=1, sample_time_step=144,
                             sample_time_length=600, step=1, train_func=None, mini_batch_size=128,
                             max_store_reward=1, infer_lower_bound=-5.0, infer_upper_bound=500, offline=False):
        """
        Run job groups with agent

        Train data generation:
            Data collection:
                Key: (Predecessor job id, Successor job id)
                Value: [ [ [] -> Queue and Cluster history (Qt, Mt, P, S), Number -> reward ], Current timestamp]
                * The data cache will be sorted based on the timestamp
            Mini batch generation:
                Key: (Predecessor job id, Successor job id)
                Value: [ [ [] -> Queue and Cluster history (Qt, Mt, P, S), Number -> reward ] -> Information per group]
                (length is customized and default value is 128)
                * For the dual-job whose reward is ready, it will be valid.
                * For other histories, it will be invalid and will not appear in the mini batch.
                * Mini batch size will be equal to or less than the actual mini batch generated.
                  (Can fill until arriving at the mini batch size)


        Train frequency:
            Call the agent for training every N seconds

        Inference frequency:
            Call the agent for inference every M seconds on all the jobs waiting for submission in the groups

        Train trigger policy:
            When train counter is equal to train frequency, trigger train function.
            When reward comes to the maximum store reward, trigger train function

        Inference trigger policy:
            When inference counter is equal to inference frequency, trigger inference function

        Exit condition:
            All the jobs in the groups has gotten the reward (? Cannot make sure all the reward will pass to agent)
            * Currently, it will stop until agent deals with all the rewards and notice the simulator to stop

        :param offline:
        :param infer_upper_bound:
        :param infer_lower_bound:
        :param max_store_reward: Maximum reward stored before one train
        :param mini_batch_size: The number of one mini dataset
        :param groups: Job groups
        :param infer_func: Inference function
        :param train_freq: Train frequency (seconds)
        :param infer_freq: Inference frequency (seconds)
        :param sample_time_step: The number of samples
        :param sample_time_length: The sample period
        :param step: Simulation running step time (seconds)
        :param train_func: Train function
        :return: None
        """
        sample_time_length = sample_time_length // int(self._mode)
        total_reward = sum([len(group) - 1 for group in groups])
        curr_reward = 0
        wait_reward_pool = []
        sample_counter = 0
        index_per_group = []
        infer_counter = 0
        # prediction_trace_counter = 0
        for group in groups:
            assert len(group) >= 2
            for i in range(1, len(group) - 1):
                group[i].prev_job = group[i - 1].job_id
                group[i].next_job = group[i + 1].job_id
            group[0].next_job = group[1].job_id
            group[len(group) - 1].prev_job = group[len(group) - 2].job_id
            self.submit_job_external([group[0]])
            index_per_group.append(1)
            self._scheduler.register_dependency(group)
        train_counter = 0
        reward_counter = 0
        num_group = 0
        for group in groups:
            num_group += len(group) - 1
        while True:
            debug_time = self._time.strftime("%Y-%m-%dT%H:%M:%S")
            if sample_counter == 0:
                sim_state = self.output_state()
                mini_batch_q = []
                mini_batch_m = []
                for item in sim_state.pending_jobs:
                    mini_batch_q.append(
                        (item.time_limit, item.nodes, int((self._time - item.submit).total_seconds()), 0))
                for _, item in sim_state.running_jobs.items():
                    mini_batch_m.append((item.job.time_limit, item.job.nodes,
                                         int((item.start - item.job.submit).total_seconds()),
                                         int(item.finish_time.total_seconds())))
                if len(self._global_history) == sample_time_step:
                    self._global_history.pop(0)
                self._global_history.append([mini_batch_q, mini_batch_m])
            sample_counter += step
            if sample_counter == sample_time_length:
                sample_counter = 0

            for i in range(0, len(groups)):
                if index_per_group[i] >= len(groups[i]):
                    continue
                if groups[i][index_per_group[i] - 1].submit >= self._time:
                    continue
                next_job = groups[i][index_per_group[i]]
                pred_job = groups[i][index_per_group[i] - 1]
                if infer_counter == infer_freq:
                    if offline:
                        if infer_func():
                            if pred_job.log is None:
                                p_wait = (self._time - pred_job.submit).total_seconds()
                                p_finish = 0
                            else:
                                p_wait = (pred_job.log.start - pred_job.submit).total_seconds()
                                p_finish = pred_job.log.finish_time.total_seconds()
                            next_job.submit = self._time
                            self._scheduler.submit([next_job])
                            wait_reward_pool.append((next_job, pred_job, int(p_wait), int(p_finish)))
                            index_per_group[i] += 1
                            logging.info(
                                "Simulator Time: {}, Successor job submission (Pred: {}, Succ: {})".format(self._time,
                                                                                                           pred_job.job_id,
                                                                                                           next_job.job_id))
                    else:
                        if pred_job.log is None:
                            p_wait = (self._time - pred_job.submit).total_seconds()
                            p_finish = 0
                        else:
                            p_wait = (pred_job.log.start - pred_job.submit).total_seconds()
                            p_finish = pred_job.log.finish_time.total_seconds()
                        if len(self._global_history) < sample_time_step:
                            continue
                        target_data = list(map(lambda x: x + [(pred_job.time_limit, pred_job.nodes, p_wait, p_finish),
                                                              (next_job.time_limit, next_job.nodes, 0, 0)],
                                               self._global_history))
                        expected_reward = infer_func([target_data])
                        logging.info(
                            f"Simulator Time: {self._time}, Complete inference. Expected Reward: {expected_reward}")
                        self._prediction_trace.append(expected_reward)
                        if infer_lower_bound <= expected_reward <= infer_upper_bound:
                            next_job.submit = self._time
                            self._scheduler.submit([next_job])
                            wait_reward_pool.append((next_job, pred_job, int(p_wait), int(p_finish), expected_reward))
                            index_per_group[i] += 1
                            logging.info(
                                "Simulator Time: {}, Successor job submission (Pred: {}, Succ: {})".format(self._time,
                                                                                                           pred_job.job_id,
                                                                                                           next_job.job_id))
            if infer_counter == infer_freq:
                infer_counter = 0
            reward_collection = self._scheduler.reward_pool
            for item in wait_reward_pool:
                if item[0].job_id in reward_collection.keys():
                    mini_batch_per_group_t = list(
                        map(lambda x: x + [(item[1].time_limit, item[1].nodes, p_wait, p_finish),
                                           (item[0].time_limit, item[0].nodes, 0, 0)], self._global_history))
                    dual_job_id_pair = (item[1].job_id, item[0].job_id)
                    self._active_reward[dual_job_id_pair] = [
                        (mini_batch_per_group_t, reward_collection[item[0].job_id][1]), None]
                    self._active_reward[dual_job_id_pair][1] = self._time
                    if not offline:
                        expected_reward = item[4]
                        actual_reward = reward_collection[item[0].job_id][1] / 3600
                        logging.info(
                            f"Simulator time: {self._time}, "
                            f"Reward Update: Expected reward: {expected_reward}, "
                            f"Actual reward: {actual_reward}, "
                            f"Error: {expected_reward - actual_reward}")
                        self._reward.append((dual_job_id_pair[0], dual_job_id_pair[1], expected_reward, actual_reward,
                                             reward_collection[item[0].job_id][2]))
                    else:
                        logging.info(
                            f"Simulator time: {self._time}, Reward Update: "
                            f"Actual reward: {reward_collection[item[0].job_id][1]}")
                    reward_counter += 1
                    curr_reward += 1
                    wait_reward_pool.remove(item)
            if train_func is not None and (reward_counter == max_store_reward):
                self._active_reward = dict((sorted(self._active_reward.items(), key=lambda x: x[1][1], reverse=True)))
                job_group_dataset = {}
                for i in range(0, min(len(self._active_reward), mini_batch_size)):
                    tmp = self._active_reward.popitem()
                    job_group_dataset[tmp[0]] = tmp[1][0]
                self._history_cache[self._time] = copy.deepcopy(job_group_dataset)
                # self._history_cache[self._time] = mini_batch
                train_func(list(job_group_dataset.values()))
                logging.info(
                    f"Simulator Time: {self._time}, Complete a train batch of {len(job_group_dataset)} samples.")
                train_counter = 0
                reward_counter = 0
            if curr_reward == total_reward:
                if train_func is not None:
                    while len(self._active_reward) != 0:
                        self._active_reward = dict(
                            (sorted(self._active_reward.items(), key=lambda x: x[1][1], reverse=True)))
                        job_group_dataset = {}
                        for i in range(0, min(len(self._active_reward), mini_batch_size)):
                            tmp = self._active_reward.popitem()
                            job_group_dataset[tmp[0]] = tmp[1][0]
                        self._history_cache[self._time] = copy.deepcopy(job_group_dataset)
                        # self._history_cache[self._time] = mini_batch
                        train_func(list(job_group_dataset.values()))
                        logging.info(
                            f"Simulator Time: {self._time}, Complete a train batch of {len(job_group_dataset)} samples.")
                return
            # train_counter += step
            infer_counter += step
            self.run_time(timedelta(seconds=step * int(self._mode)))

    def reward(self):
        return self._reward

    @property
    def prediction_trace(self):
        return self._prediction_trace

    @property
    def job_completion_logs(self):
        return self._scheduler.job_logs

    @property
    def sim_time(self):
        return self._time

    @property
    def history(self) -> {}:
        return self._history_cache

    @property
    def jobs(self):
        return self._jobs

    @jobs.setter
    def jobs(self, jobs):
        self._jobs = jobs
