import argparse
import json
import os
import subprocess
import sys

sys.path.append("../src")
sys.path.append("../src/model")

from sim import *
import torch
from moe import *
from torch.distributions import Categorical
import time
import re


def run_cmd_cluster_status(partition, clean=False):
    cmd = "squeue -a -p {} -o \"%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R %V\"".format(partition)
    print("Command: {}".format(cmd))
    if clean:
        return
    return subprocess.check_output(cmd, shell=True).decode('UTF-8')


def run_cmd_submission(project_name, script, clean=False):
    curr_dir = os.getcwd()
    os.chdir(os.path.dirname(script))
    cmd = "sbatch -A {} {}".format(project_name, script)
    print("Command: {}".format(cmd))
    if not clean:
        temp_str = subprocess.check_output(cmd, shell=True).decode('UTF-8')
        try:
            job_id = re.search(r"Submitted batch job ([0-9]+)", temp_str).group(1)
        except TypeError:
            print("Submit error! Info: {}".format(temp_str))
            exit(1)
        os.chdir(curr_dir)
    else:
        job_id = 0
    return job_id


def logtime2timedelta(time_str):
    time_str = time_str.split("-")
    days = 0
    if len(time_str) > 1:
        days = int(time_str[0])
        hour_time_str = time_str[1].split(":")
    else:
        hour_time_str = time_str[0].split(":")
    hours = 0
    minutes = 0
    seconds = 0
    if len(hour_time_str) == 3:
        hours = int(hour_time_str[0])
        minutes = int(hour_time_str[1])
        seconds = int(hour_time_str[2])
    elif len(hour_time_str) == 2:
        minutes = int(hour_time_str[0])
        seconds = int(hour_time_str[1])
    elif len(hour_time_str) == 1:
        seconds = int(hour_time_str[0])
    return timedelta(days=days, minutes=minutes, seconds=seconds, hours=hours)


def parse_output(queue_str: str):
    temp_queue_lst = queue_str.split("\n")
    temp_queue_lst = list(map(lambda x: x.split(" "), temp_queue_lst))
    title = list(filter(lambda x: x != "", temp_queue_lst[0]))
    title_index_dict = {}
    for i, name in enumerate(title):
        title_index_dict[name] = i

    pending_job_lst = []
    running_job_lst = []
    for item in temp_queue_lst[1:-1]:
        try:
            item = list(filter(lambda x: x != "", item))
            job_id = item[title_index_dict["JOBID"]]
            submit_time = item[title_index_dict["SUBMIT_TIME"]]

            # This is not accurate
            time_limit = logtime2timedelta(item[title_index_dict["TIME_LIMI"]])
            running_time = logtime2timedelta(item[title_index_dict["TIME"]])
            nodes = int(item[title_index_dict["NODES"]])
            if item[title_index_dict["STATE"]] == "PENDING":
                state = JobStatus.PENDING
            elif item[title_index_dict["STATE"]] == "RUNNING":
                state = JobStatus.RUNNING
            else:
                print("Error output job type: {}".format(item[title_index_dict["STATE"]]))
                continue
            start_time = datetime.strptime(submit_time, "%Y-%m-%dT%H:%M:%S")
            end_time = start_time + time_limit
            if state == JobStatus.RUNNING:
                start_time = datetime.now() - running_time
                end_time = start_time + time_limit
            job_inst = Job(job_id, nodes, start_time, submit_time, end_time, time_limit)
            joblog_inst = JobLog(job_inst)
            joblog_inst.run(running_time)
            joblog_inst.status = state
            if state == JobStatus.RUNNING:
                running_job_lst.append(joblog_inst)
            else:
                pending_job_lst.append(joblog_inst)
        except BaseException as e:
            print("Error parsing squeue command output: {}".format(item))
            print("Error detail: {}\n".format(e))
            continue

    return pending_job_lst, running_job_lst


def history_gen(pending_job_lst, running_job_lst):
    # [(Timelimit, Nodes, WaitingTime, RunningTime)]
    # [FirstJob: Timelimit, Nodes, WaitingTime, RunningTime]
    # [SecondJob: Timelimit, Nodes, 0, 0]
    temp_p = []
    temp_r = []
    for pending_job in pending_job_lst:
        # TODO: Is preemption valid in cluster? If it is, then finish_time should not be 0 here
        temp_p.append((int(pending_job.job.time_limit.total_seconds() / 60), pending_job.job.nodes,
                       int((datetime.now() - pending_job.job.submit).total_seconds()), 0))
    for running_job in running_job_lst:
        temp_r.append((int(running_job.job.time_limit.total_seconds() / 60), running_job.job.nodes,
                       int((datetime.now() - running_job.job.submit).total_seconds()),
                       int(running_job.finish_time.total_seconds())))
    return temp_p, temp_r


def model_setup(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model_name = checkpoint['model_name']
    model_hparams = checkpoint['model_hparams']
    model_state = checkpoint['model_state_dict']
    optimizer_name = checkpoint['optimizer_name']
    optimizer_state = checkpoint['optimizer_state_dict']

    model_pred = Predictor(model_name=model_name, hparams=model_hparams, optimizer_name=optimizer_name,
                           model_state=model_state, optimizer_state=optimizer_state,
                           base_learning_rate=1e-4, device='cpu')
    return model_pred


def model_predict(model_pred: Predictor, data_input: list, min_bound: int, max_bound: int):
    if model_pred.model_name == "moe" or model_pred.model_name == "transformer":
        np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False)

        # From numpy array to pytorch tensor
        tensor_input = torch.from_numpy(np_ndarray_input)
        # logging.debug(f"Complete inference for one job pair")
        pred_time = model_pred.inference(tensor_input).data[0].item()
    elif model_pred.model_name == "moe_policy" or model_pred.model_name == "transformer_policy":
        np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False)

        # From numpy array to pytorch tensor
        tensor_input = torch.from_numpy(np_ndarray_input)
        tensor_output = model_pred.inference(tensor_input)
        future_action = Categorical(logits=tensor_output).sample()

        if future_action.item() == 0:
            pred_time = min_bound - 1
        elif future_action.item() == 1:
            pred_time = min_bound
        else:
            print("Policy gradient sampling error: {}".format(future_action.item()))
            exit(1)
    else:
        print("Unsupported model: {}".format(model_pred.model_name))
        exit(1)

    return min_bound <= pred_time <= max_bound, pred_time


def monitor(config, history, model, partition, min_threshold, max_threshold, window_length, debug_log=None):
    # Update history
    queue_str = run_cmd_cluster_status(partition, debug_log)
    if debug_log is not None:
        with open(debug_log, "r") as f:
            queue_str = f.read()
    pending_job_lst, running_job_lst = parse_output(queue_str)
    history_p, history_r = history_gen(pending_job_lst, running_job_lst)
    history.append([history_p, history_r])
    if len(history) > window_length:
        history.pop(0)

    # Iterate all the workloads
    # Check the submission time of each job sequence
    # If the first has been submitted and second has not, use model to predict and decide
    # If the first has not been submitted and the current time is less than the submit time, continue
    # If all the jobs in the job sequence has been submitted, continue
    curr_time = datetime.now()
    complete_workload = 0
    for i in range(0, len(config["workload"])):
        item = config["workload"][i]
        if datetime.strptime(item["submit_time"], "%Y-%m-%dT%H:%M:%S") > curr_time:
            continue
        if item["job_index"] == 0:
            job_id = run_cmd_submission(config["project_name"], os.path.abspath(item["scripts"][item["job_index"]]),
                                        debug_log)
            item["job_id"] = [job_id]
            item["job_index"] += 1
        elif item["job_index"] == len(item["scripts"]):
            complete_workload += 1
            continue
        else:
            if len(history) < window_length:
                print("Current history length: {}/{}".format(len(history), window_length))
                continue
            job_in_queue = False
            p_wait = 0
            p_finish = 0
            for pending_job in pending_job_lst:
                if str(pending_job.job.job_id) == str(item["job_id"][item["job_index"] - 1]):
                    p_wait = int((datetime.now() - pending_job.job.submit).total_seconds())
                    p_finish = 0
                    job_in_queue = True
                    print("Job sequence {}: Job ID: {}, Job sequence index: {}, "
                          "Job in pending state".format(i, item["job_id"][item["job_index"] - 1], item["job_index"]))

            if not job_in_queue:
                for running_job in running_job_lst:
                    if str(running_job.job.job_id) == str(item["job_id"][item["job_index"] - 1]):
                        p_wait = int(
                            (datetime.now() - running_job.finish_time - running_job.job.submit).total_seconds())
                        p_finish = int(running_job.finish_time.total_seconds())
                        job_in_queue = True
                        print("Job sequence {}: Job ID: {}, Job sequence index: {}, "
                              "Job in running state".format(i, item["job_id"][item["job_index"] - 1],
                                                            item["job_index"]))

            target_data = list(map(lambda x: x + [(2880, 1, p_wait, p_finish),
                                                  (2880, 1, 0, 0)], history))

            is_submit = not job_in_queue
            if not is_submit:
                temp_decision, pred_time = model_predict(model, [target_data], min_threshold, max_threshold)
                print("Job sequence {}: Prediction: {}, Job sequence index: {}".format(i, pred_time, item["job_index"]))
                is_submit = temp_decision
            if is_submit:
                job_id = run_cmd_submission(config["project_name"], os.path.abspath(item["scripts"][item["job_index"]]),
                                            debug_log)
                item["job_id"].append(job_id)
                item["job_index"] += 1
    return complete_workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sample_interval", type=int, required=True,
                        help="Sample period of the cluster status (seconds)")
    parser.add_argument("-m", "--model", required=True, help="Model path")
    parser.add_argument("-partition", type=str, required=True, help="Cluster partition name")
    parser.add_argument("-min_threshold", type=float, required=True, help="Minimum threshold")
    parser.add_argument("-max_threshold", type=float, required=True, help="Maximum threshold")
    parser.add_argument("-j", "--json", type=str, required=True, help="Job submission configuration")
    parser.add_argument("-history_window", type=int, required=True, help="History window length for model")
    parser.add_argument("-out", type=str, required=True, help="Output configuration file")
    parser.add_argument("-debug_log", type=str, help="Debug log for squeue")
    args = parser.parse_args()

    with open(args.json) as f:
        config = json.load(f)

    job_number = len(config["workload"])
    cluster_history = []
    model_pred = model_setup(args.model)
    print("------------------------------------------------------------------------------")
    print("Submission tool starts...")
    print("Workload length: {}".format(job_number))
    print("------------------------------------------------------------------------------")
    print("Start sampling and prediction...")
    while job_number > 0:
        complete_workload_num = monitor(config=config, history=cluster_history, model=model_pred,
                                        partition=args.partition, min_threshold=args.min_threshold,
                                        max_threshold=args.max_threshold, window_length=args.history_window,
                                        debug_log=args.debug_log)
        job_number = len(config["workload"]) - complete_workload_num
        print("------------------------------------------------------------------------------")
        print("Wait for next sampling point. Current workload remaining: {}".format(job_number))
        time.sleep(args.sample_interval)
        print("------------------------------------------------------------------------------")

    with open(os.path.join(args.out, "config_after_running.json"), "w") as f:
        json.dump(config, f)


if __name__ == '__main__':
    main()
