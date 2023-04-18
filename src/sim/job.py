import logging
from datetime import timedelta
from enum import Enum
from datetime import datetime


class JobStatus(Enum):
    SUCCESS = 1
    RUNNING = 2
    PENDING = 3
    FAILED = 4
    NULL = 5


class Job:
    def __init__(self, job_id, nodes, start, submit, end, limit, priority=None, priority_valid=False):
        self._time_limit = limit
        if submit is not None:
            if isinstance(submit, str):
                self._submit = datetime.strptime(submit, "%Y-%m-%dT%H:%M:%S")
            else:
                self._submit = submit
        else:
            self._submit = datetime.strptime("2020-06-26T08:44:53", "%Y-%m-%dT%H:%M:%S")
        if start is not None:
            if isinstance(start, str):
                self._start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
            else:
                self._start = start
        else:
            self._start = self._submit
        if end is not None:
            if isinstance(end, str):
                self._end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
            else:
                self._end = end
        else:
            self._end = self._submit + timedelta(days=2)
        
        if isinstance(self._time_limit, timedelta):
            time_limit_temp = self._time_limit
        else:
            time_limit_temp = timedelta(minutes=self._time_limit)
        if self._end - self._start > time_limit_temp:
            logging.warning("Error Job information: Duration > Maximum Time Limit, Job ID: {}".format(job_id))
            self._end = self._start + time_limit_temp
        self._nodes = nodes
        self._job_id = job_id
        self._priority = priority
        self._priority_valid = priority_valid
        self._extra_priority = 0
        self._prev = None
        self._prev_end = None
        self._next = None
        self.log = None
        self.backfill_time = datetime.strptime("2077-01-01T01:00:00", "%Y-%m-%dT%H:%M:%S")

    @property
    def wait_time(self):
        return self._start - self._submit

    @property
    def early_priority(self):
        return self._extra_priority

    @property
    def priority(self):
        return self._priority

    @property
    def time_limit(self):
        return self._time_limit

    @property
    def nodes(self):
        return self._nodes

    @property
    def duration(self):
        return self._end - self._start

    @property
    def submit(self):
        return self._submit

    @early_priority.setter
    def early_priority(self, priority):
        self._extra_priority = priority

    @submit.setter
    def submit(self, submit):
        self._submit = submit

    @priority.setter
    def priority(self, priority):
        self._priority = priority

    @property
    def job_id(self):
        return self._job_id

    @property
    def priority_from_log(self):
        return self._priority_valid

    @property
    def prev_job_end(self):
        return self._prev_end

    @prev_job_end.setter
    def prev_job_end(self, end_time):
        self._prev_end = end_time

    @property
    def prev_job(self):
        return self._prev

    @property
    def next_job(self):
        return self._next

    @prev_job.setter
    def prev_job(self, prev_job):
        self._prev = prev_job

    @next_job.setter
    def next_job(self, next_job):
        self._next = next_job

    def info_str(self):
        return "ID: {:<10} Priority: {:<15.4f} Submit: {:<25} Duration: {:<15} Time Limit: {:<10} Nodes: {}".format(
            self._job_id,
            self._priority,
            str(self._submit),
            self.duration.total_seconds(),
            self._time_limit,
            self._nodes)


class JobLog:

    def __init__(self, job):
        self._start = None
        self._end = None
        self._original_end = None
        self._job = job
        self._status = JobStatus.NULL
        self._running_time = timedelta(seconds=0)
        self._nodes = []

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def job(self):
        return self._job

    @property
    def status(self):
        return self._status

    @property
    def finish_time(self):
        return self._running_time

    @status.setter
    def status(self, status):
        self._status = status

    @start.setter
    def start(self, start):
        self._start = start
        self._original_end = self._start + timedelta(minutes=self._job.time_limit)

    @end.setter
    def end(self, end):
        self._end = end

    @property
    def nodes(self):
        return self._nodes

    @property
    def original_end(self):
        return self._original_end

    def add_nodes(self, node):
        self._nodes.append(node)

    def run(self, time):
        if self._running_time + time >= self._job.duration:
            self._running_time = self._job.duration
            return True
        else:
            self._running_time += time
            return False

    def info_str(self):
        return "ID: {:<10} Priority: {:<15.4f} Submit: {:<25} Start: {:<25} End: {:<25} Backfill: {:<25} Nodes: {}".format(
            self._job.job_id,
            self._job.priority,
            str(self._job.submit),
            str(self._start),
            str(self._end),
            str(self._job.backfill_time),
            self._nodes)
