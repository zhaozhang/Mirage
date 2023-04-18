import queue
from sim import job
import logging
from datetime import timedelta


class Cluster:
    def __init__(self, nodes, cluster_start_time, mode):
        self._processors = dict(zip([i for i in range(0, nodes)], [None for _ in range(0, nodes)]))
        self._job_info = {}
        self._num_proc = nodes
        self._execution_queue = queue.PriorityQueue()
        self._available_proc = [i for i in range(0, nodes)]
        self._time = cluster_start_time
        self._mode = mode

    @DeprecationWarning
    def run_event(self) -> []:
        """
        TODO: Useless part for dynamic pending queue
        """
        if self._execution_queue.empty():
            logging.debug("No jobs in the cluster. Event run fails.")
            return []
        complete_jobs = [self._execution_queue.get()]
        running_time = complete_jobs[0][0] - self._time
        self._time = complete_jobs[0][0]
        while True:
            if self._execution_queue.empty():
                break
            complete_job_tmp = self._execution_queue.get()
            if complete_job_tmp[0] > self._time:
                self._execution_queue.put(complete_job_tmp)
                break
            else:
                complete_jobs.append(complete_job_tmp)
        for key, value in self._job_info.items():
            value.run(running_time)
        complete_jobs_log = []
        for item in complete_jobs:
            complete_job_log = self._job_info.pop(item[1])
            for proc in complete_job_log.nodes:
                self._processors[proc] = None
                self._available_proc.append(proc)
            complete_jobs_log.append(complete_job_log)
        logging.info(
            "Run cluster successfully. Time: {0}, Running Time: {1}".format(self._time.strftime("%m/%d/%Y, %H:%M:%S"),
                                                                            running_time))
        return complete_jobs_log

    def run_time(self, time) -> []:
        self._time += time
        logging.debug(
            "Run cluster successfully. Time: {0}, Running Time: {1}".format(self._time.strftime("%m/%d/%Y, %H:%M:%S"),
                                                                            time))

        """
        if self._execution_queue.empty():
            return []
        possible_complete_job = self._execution_queue.get()
        complete_jobs = []
        if possible_complete_job[0] <= self._time:
            complete_jobs.append(possible_complete_job[1])
            while True:
                if self._execution_queue.empty():
                    break
                complete_job_tmp = self._execution_queue.get()
                if complete_job_tmp[0] > self._time:
                    self._execution_queue.put(complete_job_tmp)
                    break
                else:
                    complete_jobs.append(complete_job_tmp[1])
        else:
            self._execution_queue.put(possible_complete_job)
        """
        complete_jobs = []
        for key, value in self._job_info.items():
            if value.run(time):
                complete_jobs.append(key)
        complete_jobs_log = []
        for item in complete_jobs:
            complete_job_log = self._job_info.pop(item)
            complete_job_log.status = job.JobStatus.SUCCESS
            for proc in complete_job_log.nodes:
                self._processors[proc] = None
                self._available_proc.append(proc)
            complete_jobs_log.append(complete_job_log)
        return complete_jobs_log

    def submit_job(self, new_job):
        if len(self._available_proc) < new_job.nodes:
            logging.warning("Job submission failure. Job ID: " + new_job.job_id)
            return False
        job_log = job.JobLog(new_job)
        new_job.log = job_log
        job_log.status = job.JobStatus.RUNNING
        job_log.start = self._time
        job_log.end = self._time + timedelta(seconds=(new_job.duration.total_seconds() // int(self._mode)) * int(self._mode))
        self._job_info[new_job.job_id] = job_log
        # self._execution_queue.put((job_log.end, new_job.job_id))
        for _ in range(0, new_job.nodes):
            proc = self._available_proc.pop()
            job_log.add_nodes(proc)
            self._processors[proc] = new_job.job_id
        return True

    def reconfig(self, job_infos, cluster_time):
        # TODO: Pre-set the processor states, job info dictionary, execution_queue, and current cluster time
        self._processors = dict(zip([i for i in range(0, self._num_proc)], [None for _ in range(0, self._num_proc)]))
        self._job_info = job_infos
        self._available_proc = [i for i in range(0, self._num_proc)]
        self._time = cluster_time
        self._execution_queue = queue.PriorityQueue()
        for key, value in job_infos.items():
            for node in value.nodes:
                self._processors[node] = key
                self._available_proc.remove(node)
        for key, value in job_infos.items():
            self._execution_queue.put((value.end, key))
        return

    @property
    def available_proc(self):
        return len(self._available_proc)

    @property
    def proc_status(self):
        return self._processors.items()

    @property
    def job_info(self):
        return self._job_info

    @property
    def num_proc(self):
        return self._num_proc
