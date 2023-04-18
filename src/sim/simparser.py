import json
import pickle
import numpy as np
from sim import job
from datetime import datetime
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class SimParser:

    @staticmethod
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

    @staticmethod
    def parse_job(log, pivot=None, job_filter=None):
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
            if job_filter is not None:
                if any(True for filter_item in job_filter if filter_item[0] <= submit_time < filter_item[1]):
                    continue
            if (end_time - start_time).total_seconds() > (int(job_info[index_dict["TimelimitR"]]) * 60):
                end_time = start_time + timedelta(seconds=int(job_info[index_dict["TimelimitR"]]) * 60)
            new_job = job.Job(job_info[index_dict["JobID"]], int(job_info[index_dict["NNodes"]]), start_time,
                              submit_time, end_time, int(job_info[index_dict["TimelimitR"]]))
            if pivot is not None:
                if new_job.duration.total_seconds() / (new_job.time_limit * 60) < pivot:
                    continue
            job_dict[job_info[index_dict["JobID"]]] = new_job
        return job_dict

    @staticmethod
    def parse_pickle_job(log, time_limit_gen):
        with open(log, "rb") as f:
            content = pickle.load(f)
        job_id = 0
        job_dict = {}
        total_duration = []
        for item in content:
            total_duration.append(item["duration_in_seconds"])
        total_duration = np.array(total_duration).reshape(-1, 1)
        pred_y = time_limit_gen(total_duration)

        for i in range(0, len(content)):
            start_time = content[i]["submission_timestamp"]
            submit_time = content[i]["submission_timestamp"]
            end_time = start_time + timedelta(seconds=content[i]["duration_in_seconds"])
            if pred_y[i] > 2880:
                pred_y[i] = 2880
            new_job = job.Job(str(job_id), content[i]["num_gpus"], start_time, submit_time, end_time, int(pred_y[i]))
            job_dict[str(job_id)] = new_job
            job_id += 1
        return job_dict

    @staticmethod
    def linear_reg_time_limit(jobs):
        duration = []
        timelimit = []
        for key, value in jobs.items():
            duration.append([value.duration.total_seconds()])
            timelimit.append(value.time_limit)
        duration = np.array(duration).reshape(-1, 1)
        timelimit = np.array(timelimit)
        X_train, X_test, y_train, y_test = train_test_split(duration, timelimit, test_size=0.33, random_state=0)
        regression = RandomForestRegressor(max_depth=12, random_state=100)
        regression.fit(X_train, y_train)
        return regression

    @staticmethod
    def load_slurm_config(config_file):
        with open(config_file) as f:
            return json.load(f)

    @staticmethod
    def load_backfill_config(config_file):
        with open(config_file) as f:
            return json.load(f)
