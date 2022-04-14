import json
import job
import logging
from datetime import datetime
from datetime import timedelta


@DeprecationWarning
def parseJobCompleteLog(log, priority_dict):
    with open(log) as f:
        content = f.read()
    job_log = content.split("\n")
    index_lst = list(map(lambda x: x.strip(), job_log[0].split("|")))
    index_dict = dict(zip(index_lst, [i for i in range(0, len(index_lst))]))
    job_lst = []
    for row in job_log:
        one_job = list(map(lambda x: x.strip(), row.split("|")))
        job_lst.append(job.Job(one_job[index_dict["JobID"]], one_job[index_dict["NNodes"]],
                               one_job[index_dict["Start"]], one_job[index_dict["Submit"]],
                               one_job[index_dict["End"]], one_job[index_dict["Timelimit"]],
                               priority_dict[one_job[index_dict["JobID"]]]))
    return job_lst


def parse_job(log):
    with open(log) as f:
        content = f.read()
    job_log = content.split("\n")
    index_lst = list(filter(lambda x: x != "", job_log[0].split(" ")))
    index_dict = dict(zip(index_lst, [i for i in range(0, len(index_lst))]))
    job_dict = {}
    for row in job_log[2:-1]:
        job_info = list(filter(lambda x: x != "", row.split(" ")))
        start_time = datetime.strptime(job_info[index_dict["Start"]], "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.strptime(job_info[index_dict["End"]], "%Y-%m-%dT%H:%M:%S")
        submit_time = datetime.strptime(job_info[index_dict["Submit"]], "%Y-%m-%dT%H:%M:%S")
        if (end_time - start_time).total_seconds() > (int(job_info[index_dict["TimelimitR"]]) * 60):
            end_time = start_time + timedelta(seconds=int(job_info[index_dict["TimelimitR"]]) * 60)
        new_job = job.Job(job_info[index_dict["JobID"]], int(job_info[index_dict["NNodes"]]), start_time,
                          submit_time, end_time, int(job_info[index_dict["TimelimitR"]]))
        job_dict[job_info[index_dict["JobID"]]] = new_job
    return job_dict


def load_slurm_config(config_file):
    with open(config_file) as f:
        return json.load(f)


def load_backfill_config(config_file):
    with open(config_file) as f:
        return json.load(f)
