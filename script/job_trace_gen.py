import argparse
from datetime import datetime
from datetime import timedelta
import csv
import os
import json
import logging


class Policy:

    def update(self, index_dict, job):
        return

    def get_jobs(self):
        return


class AvgPolicy(Policy):
    def __init__(self, jobs, index_dict):
        self.jobs = sorted(jobs, key=lambda x: x[index_dict["Submit"]])
        self.jobs_avg_queue_time = [[0, 0] for _ in range(0, len(jobs))]
        self.index_dict = index_dict

    def update(self, index_dict, job):
        start_time = datetime.strptime(job[index_dict["Start"]], "%Y-%m-%dT%H:%M:%S")
        submit_time = datetime.strptime(job[index_dict["Submit"]], "%Y-%m-%dT%H:%M:%S")
        if (start_time - submit_time).total_seconds() > 48 * 3600:
            return
        for i in range(0, len(self.jobs)):
            start_prev = self.jobs[i][self.index_dict["Submit"]] >= job[index_dict["Start"]]
            submit_prev = self.jobs[i][self.index_dict["Submit"]] >= job[index_dict["Submit"]]
            if start_prev and submit_prev:
                self.jobs_avg_queue_time[i][0] += (start_time - submit_time).total_seconds()
                self.jobs_avg_queue_time[i][1] += 1

    def get_jobs(self):
        for i in range(0, len(self.jobs)):
            time = datetime.strptime(self.jobs[i][self.index_dict["Submit"]], "%Y-%m-%dT%H:%M:%S")
            time = time - timedelta(
                seconds=int(self.jobs_avg_queue_time[i][0] / self.jobs_avg_queue_time[i][1]))
            time = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.jobs[i][self.index_dict["Submit"]] = time
        return self.jobs


class InitPolicy(Policy):

    def __init__(self, jobs, index_dict):
        self.jobs = jobs
        self.index_dict = index_dict

    def get_jobs(self):
        return self.jobs


class InitialInsertPolicy(Policy):

    def __init__(self, start_time, end_time, freq):
        time_interval = timedelta(days=freq)
        curr_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
        self.jobs = []
        job_dict = {"JobID": 1000000, "Submit": "", "TimelimitR": 2880, "Start": "2019-03-12T09:01:01",
                    "End": "2019-03-14T09:01:01", "ExitCode": "0:0", "NNodes": 1}
        self.index_dict = dict(zip(job_dict.keys(), [i for i in range(0, len(job_dict.keys()))]))
        while curr_time <= end_time:
            job_dict["Submit"] = curr_time.strftime("%Y-%m-%dT%H:%M:%S")
            self.jobs += [list(job_dict.values())]
            curr_time += time_interval
            job_dict["JobID"] += 1

    def get_jobs(self):
        return self.jobs


def load_insert_json(json_file):
    with open(json_file) as f:
        insert_job = json.load(f)["job"]
    index_dict = {}
    for i in range(0, len(insert_job[0].keys())):
        index_dict[list(insert_job[0].keys())[i]] = i
    job_real_lst = []
    job_future_lst = []
    for i in range(0, len(insert_job)):
        tmp_job = list(insert_job[i].values())
        if tmp_job[index_dict["real"]]:
            job_real_lst.append(tmp_job)
        else:
            job_future_lst.append(tmp_job)
    return index_dict, job_real_lst, job_future_lst


def sim_job_gen(cpu_per_node, index_dict, row):
    tmp_str = "sim_job("
    tmp_str += "job_id={}, ".format(row[index_dict["JobID"]])
    tmp_str += "submit=\"{}\", ".format(row[index_dict["Submit"]].replace("T", " "))
    tmp_str += "wclimit={}L, ".format(row[index_dict["TimelimitR"]])
    start_time = datetime.strptime(row[index_dict["Start"]], "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.strptime(row[index_dict["End"]], "%Y-%m-%dT%H:%M:%S")
    elapse = int((end_time - start_time).total_seconds())
    if ((elapse / 60) > float(row[index_dict["TimelimitR"]])) and row[index_dict["ExitCode"]] == "0:0":
        elapse = int(row[index_dict["TimelimitR"]]) * 60
    tmp_str += "duration={}L, ".format(elapse)
    tmp_str += "shared=0L, "
    tmp_str += "tasks={}L, ".format(int(row[index_dict["NNodes"]]) * cpu_per_node)
    tmp_str += "tasks_per_node={}L".format(cpu_per_node)
    tmp_str += ")"
    return tmp_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job", required=True, help="Job log")
    parser.add_argument("-i", "--insert", help="Insert job json file", default=None)
    parser.add_argument("-o", "--output_script", default="./test.R", help="Trace output location")
    parser.add_argument("-policy", type=str, help="Policy type")
    parser.add_argument("-t", "--output_trace", default="./test.trace", help="Trace output location")
    args = parser.parse_args()

    insert = False
    insert_policy = Policy()
    job_exist_lst = []
    if args.insert is not None:
        index_dict_insert, job_exist_lst, job_insert_lst = load_insert_json(args.insert)
        if args.policy == "avg":
            insert_policy = AvgPolicy(job_insert_lst, index_dict_insert)
        elif args.policy == "initInsert":
            insert_policy = InitialInsertPolicy("2019-11-11T09:00:00", "2021-08-19T22:25:03", 7)
            job_exist_lst = []
        else:
            insert_policy = InitPolicy(job_insert_lst, index_dict_insert)
        insert = True
    cpu_per_node = 12

    index_dict = {}
    count = 0
    if insert:
        job_lst = [sim_job_gen(cpu_per_node, insert_policy.index_dict, job) for job in job_exist_lst]
        for job in job_exist_lst:
            insert_policy.update(insert_policy.index_dict, job)
    else:
        job_lst = []
    with open(args.job) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if count == 0:
                for i in range(0, len(row)):
                    index_dict[row[i]] = i
                count = 1
                continue
            if insert:
                insert_policy.update(index_dict, row)
            tmp_str = sim_job_gen(cpu_per_node, index_dict, row)
            job_lst.append(tmp_str)

    if insert:
        job_insert_lst = insert_policy.get_jobs()
        for job in job_insert_lst:
            job_lst.append(sim_job_gen(cpu_per_node, insert_policy.index_dict, job))

    with open(args.output_script, "w") as f:
        f.write("# This file is automatically generated\n\n")
        f.write(
            "library(RSlurmSimTools)\ntop_dir <- \"{}\"\nprint(top_dir)\nsetwd(top_dir)\n\n".format(
                os.path.dirname(args.output_trace)))
        f.write("trace <- list(\n")
        for i in range(0, len(job_lst)):
            if i == len(job_lst) - 1:
                f.write("\t" + job_lst[i] + "\n")
            else:
                f.write("\t" + job_lst[i] + ",\n")
        f.write(")\n")
        f.write("trace <- do.call(rbind, lapply(trace, data.frame))\n")
        f.write("write_trace(file.path(top_dir,\"{}\"), trace)\n".format(os.path.basename(args.output_trace)))


if __name__ == '__main__':
    main()
