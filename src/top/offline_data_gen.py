import argparse
import datetime
import json
import logging
import os
import sys
import pickle

import numpy

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)
from sim import *
import sim
import ray
import math


def pre_simulate(simulator_start_time, workload_start_time, workload_end_time, slurm_config, backfill_config,
                 workload_path, warmup_len, regr_model, queue_time_len, pivot, filter_kernel, node):
    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    if regr_model is None:
        jobs = SimParser.parse_job(workload_path, pivot, filter_kernel)
    else:
        jobs = SimParser.parse_pickle_job(workload_path, regr_model.predict)

    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs

    slurm_simulator.init_scheduler(88, slurm_config, simulator_start_time, backfill_config, queue_time_len)

    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)
    # ---------------------------------------------------------------------------------------------------------------

    time = workload_start_time + timedelta(days=warmup_len)
    job_1 = Job("1000000", node, None, time, None, 2880, 6000, False)
    slurm_simulator.submit_job_external([job_1])
    slurm_simulator.run_end(job_1.job_id)
    job_end_time = slurm_simulator.sim_time + job_1.duration - timedelta(minutes=1)
    # slurm_simulator.run_time(timedelta(days=2))
    return time + timedelta(minutes=1), job_end_time, int(
        slurm_simulator.avg_queue_time() / 60), slurm_simulator.waiting_queue_size


def simulate_per_sample(sample_time, simulator_start_time, workload_start_time, workload_end_time, slurm_config,
                        backfill_config, workload_path, warmup_len, regr_model, early_age, sample_time_window, pivot,
                        filter_kernel, node):
    def infer_wrapper(simulator_time):
        if simulator_time == sample_time:
            return True
        return False

    def train_wrapper(batch, cache):
        cache.append(batch)
        return

    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    if regr_model is None:
        jobs = SimParser.parse_job(workload_path, pivot, filter_kernel)
    else:
        jobs = SimParser.parse_pickle_job(workload_path, regr_model.predict)
    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs

    slurm_simulator.init_scheduler(88, slurm_config, simulator_start_time, backfill_config)

    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)
    # ---------------------------------------------------------------------------------------------------------------

    time = workload_start_time + timedelta(days=warmup_len)
    job_1 = Job("1000000", node, None, time, None, 2880, 6000, False)
    job_2 = Job("1000001", node, None, time, None, 2880, 6000, False)
    batch_cache = []

    slurm_simulator.run_group_dependency([[job_1, job_2]],
                                         infer_func=lambda: infer_wrapper(slurm_simulator.sim_time),
                                         train_func=lambda x: train_wrapper(x, batch_cache),
                                         sample_time_length=600,
                                         train_freq=10,
                                         infer_freq=1,
                                         max_store_reward=1,
                                         sample_time_step=sample_time_window,
                                         infer_lower_bound=-600,
                                         infer_upper_bound=600,
                                         early_age=early_age,
                                         offline=True)
    return batch_cache


@ray.remote
def simulate_per_thread(init_time, i, slurm_config, backfill_config, num_probe, start_time_interval, workload_path,
                        workload_len, warmup_len, early_age, baseline_avg_queue_time_len, sample_time_window, pivot,
                        mode="default", regr_model=None, quantile_model=None, filter_kernel=None, node=1):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.ERROR)
    data = []
    simulator_start_time = init_time + timedelta(hours=i * start_time_interval)
    workload_start_time = simulator_start_time
    workload_end_time = workload_start_time + timedelta(days=workload_len)
    lower_bound, upper_bound, avg_time, waiting_queue_size = pre_simulate(simulator_start_time, workload_start_time,
                                                                          workload_end_time,
                                                                          slurm_config, backfill_config, workload_path,
                                                                          warmup_len, regr_model,
                                                                          baseline_avg_queue_time_len, pivot,
                                                                          filter_kernel, node)
    if mode == "baseline_avg":
        sample_point = [upper_bound - timedelta(minutes=avg_time)]
    elif mode == "baseline_reactive":
        sample_point = [upper_bound]
    elif mode == "baseline_quantile_regr":
        if quantile_model is None:
            sample_point = [upper_bound]
        else:
            test = [(avg_time / 60) * waiting_queue_size]
            test = numpy.array(test).reshape(-1, 1)
            pred_value = int(quantile_model.predict(test)[0] * 60)
            if pred_value < 0:
                pred_value = 0
            elif pred_value > 2880:
                pred_value = 2880
            sample_point = [upper_bound - timedelta(minutes=pred_value)]
    else:
        interval = (upper_bound - lower_bound).total_seconds() / (num_probe - 1.0)
        sample_point = [(lower_bound + timedelta(seconds=interval * j)).replace(second=0) for j in range(0, num_probe)]
    for point in sample_point:
        data.append(
            simulate_per_sample(point, simulator_start_time, workload_start_time, workload_end_time,
                                slurm_config, backfill_config, workload_path, warmup_len, regr_model, early_age,
                                sample_time_window, pivot, filter_kernel, node)[0][0])
    return data, avg_time, i, waiting_queue_size


def simulate_single(init_time, i, slurm_config, backfill_config, num_probe, start_time_interval, workload_path,
                    workload_len, warmup_len, early_age, baseline_avg_queue_time_len, sample_time_window, pivot,
                    mode="default", regr_model=None, quantile_model=None, filter_kernel=None, node=1):
    data = []
    simulator_start_time = init_time + timedelta(hours=i * start_time_interval)
    workload_start_time = simulator_start_time
    workload_end_time = workload_start_time + timedelta(days=workload_len)
    lower_bound, upper_bound, avg_time, waiting_queue_size = pre_simulate(simulator_start_time, workload_start_time,
                                                                          workload_end_time,
                                                                          slurm_config, backfill_config, workload_path,
                                                                          warmup_len,
                                                                          regr_model, baseline_avg_queue_time_len,
                                                                          pivot, filter_kernel, node)
    if mode == "baseline_avg":
        sample_point = [upper_bound - timedelta(minutes=avg_time)]
    elif mode == "baseline_reactive":
        sample_point = [upper_bound]
    elif mode == "baseline_quantile_regr":
        if quantile_model is None:
            sample_point = [upper_bound]
        else:
            test = [(avg_time / 60) * waiting_queue_size]
            test = numpy.array(test).reshape(-1, 1)
            pred_value = int(quantile_model.predict(test)[0] * 60)
            sample_point = [upper_bound - timedelta(minutes=pred_value)]
    else:
        interval = (upper_bound - lower_bound).total_seconds() / (num_probe - 1.0)
        sample_point = [(lower_bound + timedelta(seconds=interval * j)).replace(second=0) for j in range(0, num_probe)]
    for point in sample_point:
        data.append(
            simulate_per_sample(point, simulator_start_time, workload_start_time, workload_end_time,
                                slurm_config, backfill_config, workload_path, warmup_len, regr_model, early_age,
                                sample_time_window, pivot, filter_kernel, node)[0][0])
    return data, avg_time


def simulate(num_probe, start_time_interval, parallel, num_sample_all, workload_name, sim_init_time, workload_len,
             warmup_len, output_dir, file_split, file_num, cpu_cores, early_age, slurm_config,
             backfill_config, quantile_model, baseline_avg_queue_time_len, sample_time_window, pivot, mode="default",
             regr_model=None, file_init_count=0, start_index=0, filter_kernel=None, node=1):
    sample_data = {}
    sample_avg = {}
    sample_queue_size = {}
    file_count = file_init_count
    remaining_samples = num_sample_all
    if not parallel:
        for i in range(start_index, start_index + num_sample_all):
            print(f"Execute job {i + 1}/{num_sample_all}...")
            sim_start_time = sim_init_time + timedelta(hours=i * start_time_interval)
            if filter_kernel is not None:
                if any(True for filter_item in filter_kernel if filter_item[0] <= sim_start_time < filter_item[1]):
                    print(f"Ignore sample points within the filter time range (Sample start time: {sim_start_time})")
                    remaining_samples -= 1
                    continue
            data, avg = simulate_single(init_time=sim_init_time,
                                        i=i,
                                        slurm_config=slurm_config,
                                        backfill_config=backfill_config,
                                        num_probe=num_probe,
                                        start_time_interval=start_time_interval,
                                        workload_path=os.path.join("../workload", workload_name),
                                        workload_len=workload_len,
                                        warmup_len=warmup_len,
                                        early_age=early_age,
                                        baseline_avg_queue_time_len=baseline_avg_queue_time_len,
                                        sample_time_window=sample_time_window,
                                        pivot=pivot,
                                        mode=mode,
                                        regr_model=regr_model,
                                        quantile_model=quantile_model,
                                        filter_kernel=filter_kernel,
                                        node=node)
            sample_data[i] = data
            sample_avg[i] = avg
        print("-----------------------")
    else:
        file = os.listdir("../workload")
        file = list(filter(lambda x: x != workload_name, file))
        file = list(map(lambda x: "/workload/" + x, file))
        if cpu_cores == -1:
            ray.init(runtime_env={"py_modules": [sim], "working_dir": "../",
                                  "excludes": ["/test/", "/data/", "/misc/"] + file})
        else:
            ray.init(runtime_env={"py_modules": [sim], "working_dir": "../",
                                  "excludes": ["/test/", "/data/", "/misc/"] + file},
                     num_cpus=cpu_cores)
        rpc = []
        for i in range(start_index, num_sample_all + start_index):
            print(f"Submitting job {i + 1}/{num_sample_all}...")
            sim_start_time = sim_init_time + timedelta(hours=i * start_time_interval)
            if filter_kernel is not None:
                if any(True for filter_item in filter_kernel if filter_item[0] <= sim_start_time < filter_item[1]):
                    print(f"Ignore sample points within the filter time range (Sample start time: {sim_start_time})")
                    remaining_samples -= 1
                    continue
            rpc.append(
                simulate_per_thread.remote(sim_init_time, i, slurm_config, backfill_config, num_probe,
                                           start_time_interval, os.path.join("./workload", workload_name), workload_len,
                                           warmup_len, early_age, baseline_avg_queue_time_len, sample_time_window,
                                           pivot, mode, regr_model, quantile_model, filter_kernel, node))
        print("-----------------------")
        finish_count = 0
        total_finish_time = timedelta(seconds=0)
        while len(rpc):
            wait_time = datetime.today()
            done_id, rpc = ray.wait(rpc)
            finish_count += 1
            res = ray.get(done_id[0])
            # Only one group of dependency job
            sample_data[res[2]] = res[0]
            sample_avg[res[2]] = res[1]
            sample_queue_size[res[2]] = res[3]
            total_finish_time += datetime.today() - wait_time
            avg_complete_sec = total_finish_time.total_seconds() / finish_count
            expected_running_time = avg_complete_sec * (num_sample_all - finish_count)
            print(
                f"Job Finish {finish_count}/{num_sample_all}, "
                f"Average finish time: "
                f"{'{}:{}:{}'.format(int(avg_complete_sec // 3600), int(avg_complete_sec % 3600 // 60), int(avg_complete_sec % 60))}, "
                f"Expected remaining time: "
                f"{'{}:{}:{}'.format(int(expected_running_time // 3600), int(expected_running_time % 3600 // 60), int(expected_running_time % 60))}",
                flush=True)
            if len(sample_data) == (num_sample_all / file_num) and file_split:
                with open(os.path.join(output_dir, f"data_{file_count}.pickle"), "wb") as f:
                    pickle.dump(sample_data, f)
                sample_data.clear()
                file_count += 1
    if len(sample_data) != 0:
        if not file_split:
            file_count = -1
        with open(os.path.join(output_dir, f"data_{file_count + 1}.pickle"), "wb") as f:
            pickle.dump(sample_data, f)
        sample_data.clear()
    with open(os.path.join(output_dir, f"data_avg_{file_init_count}.pickle"), "wb") as f:
        pickle.dump(sample_avg, f)
    with open(os.path.join(output_dir, f"data_queue_size_{file_init_count}.pickle"), "wb") as f:
        pickle.dump(sample_queue_size, f)
    print(f"Remaining samples: {remaining_samples}/{num_sample_all}")
    print("-----------------------")
    ray.shutdown()


def main():
    prev_time = datetime.today()
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("-parallel", action="store_true", default=False)
    parser_arg.add_argument("-num_samples", type=int, default=2)
    parser_arg.add_argument("-num_probe", type=int, default=5)
    parser_arg.add_argument("-interval", "--start_time_interval", type=int, default=6)
    parser_arg.add_argument("-baseline", type=str, default="default")
    parser_arg.add_argument("-workload", "--workload_name", type=str, default="filtered-longhorn-v100.log")
    parser_arg.add_argument("-start_time", type=lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"),
                            default="2021-03-01T00:00:00")
    parser_arg.add_argument("-workload_len", type=int, default=5)
    parser_arg.add_argument("-od", "--output_dir", default="./")
    parser_arg.add_argument("-cd", "--config_dir", default="../test/test_data")
    parser_arg.add_argument("-warmup_len", type=int, default=2)
    parser_arg.add_argument("-file_split", action="store_true", default=False)
    parser_arg.add_argument("-file_num", type=int, default=4)
    parser_arg.add_argument("-philly_tl", "--philly_time_limit", default=None)
    parser_arg.add_argument("-philly", action="store_true", default=False)
    parser_arg.add_argument("-ref_trace", default="../workload/filtered-frontera-rtx.log")
    parser_arg.add_argument("-cpu_cores", type=int, default=-1)
    parser_arg.add_argument("-early_age", action="store_true", default=False)
    parser_arg.add_argument("-ray_reset", action="store_true", default=False)
    parser_arg.add_argument("-quantile_model", default=None)
    parser_arg.add_argument("-baseline_avg_num", type=int, default=100)
    parser_arg.add_argument("-window", type=int, default=144)
    parser_arg.add_argument("-job_duration_pivot", type=float, default=None)
    parser_arg.add_argument("-start_index", type=int, default=0)
    parser_arg.add_argument("-job_filter", type=str, default=None)
    parser_arg.add_argument("-node", type=int, default=1)
    args = parser_arg.parse_args()

    if args.job_filter is not None:
        with open(args.job_filter) as f:
            job_filter = json.load(f)["filter_range"]
        job_filter = list(map(lambda x: list(map(lambda y: datetime.strptime(y, "%Y-%m-%dT%H:%M:%S"), x)), job_filter))
    else:
        job_filter = None

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.ERROR)
    print("--------------------------------------------------------------------------------------------------------")
    print("Simulation starts...")
    os.makedirs(args.output_dir, exist_ok=True)
    print("-----------------------")
    regr = None
    quantile_model = None
    if args.philly:
        print("Fit model for time limit... Start")
        if args.philly_time_limit is None:
            ref_jobs = SimParser.parse_job(args.ref_trace)
            regr = SimParser.linear_reg_time_limit(ref_jobs)
            with open(os.path.join(args.output_dir, "model_time_limit.pickle"), "wb") as f:
                pickle.dump(regr, f)
        else:
            print("Loading time limit model...")
            with open(args.philly_time_limit, "rb") as f:
                regr = pickle.load(f)
        print("Fit model for time limit... End")
        print("-----------------------")
    if args.baseline == "baseline_quantile_regr":
        if args.quantile_model is not None:
            print("Loading Quantile Regression model...")
            with open(args.quantile_model, "rb") as f:
                quantile_model = pickle.load(f)

    slurm_config = simparser.SimParser.load_slurm_config(os.path.join(args.config_dir, "slurm_config.json"))
    backfill_config = simparser.SimParser.load_backfill_config(os.path.join(args.config_dir, "backfill_config.json"))
    if args.ray_reset and args.file_split:
        file_step = math.ceil(args.num_samples / args.file_num)
        for i in range(0, args.file_num):
            print("-----------------------")
            print(f"Partially Simulate {i + 1}/{args.file_num}...")
            sim_init_time = args.start_time + timedelta(hours=i * file_step * args.start_time_interval)
            if i == (args.file_num - 1):
                file_step = args.num_samples
            simulate(num_probe=args.num_probe,
                     start_time_interval=args.start_time_interval,
                     parallel=args.parallel,
                     num_sample_all=file_step,
                     workload_name=args.workload_name,
                     sim_init_time=sim_init_time,
                     workload_len=args.workload_len,
                     warmup_len=args.warmup_len,
                     output_dir=args.output_dir,
                     file_split=args.file_split,
                     file_num=1,
                     cpu_cores=args.cpu_cores,
                     early_age=args.early_age,
                     mode=args.baseline,
                     slurm_config=slurm_config,
                     backfill_config=backfill_config,
                     regr_model=regr,
                     file_init_count=i,
                     quantile_model=quantile_model,
                     baseline_avg_queue_time_len=args.baseline_avg_num,
                     sample_time_window=args.window,
                     pivot=args.job_duration_pivot,
                     start_index=args.start_index,
                     filter_kernel=job_filter,
                     node=args.node)
            args.num_samples -= file_step
        print("-----------------------")
    else:
        simulate(num_probe=args.num_probe,
                 start_time_interval=args.start_time_interval,
                 parallel=args.parallel,
                 num_sample_all=args.num_samples,
                 workload_name=args.workload_name,
                 sim_init_time=args.start_time,
                 workload_len=args.workload_len,
                 warmup_len=args.warmup_len,
                 output_dir=args.output_dir,
                 file_split=args.file_split,
                 file_num=args.file_num,
                 cpu_cores=args.cpu_cores,
                 early_age=args.early_age,
                 mode=args.baseline,
                 slurm_config=slurm_config,
                 backfill_config=backfill_config,
                 regr_model=regr,
                 quantile_model=quantile_model,
                 baseline_avg_queue_time_len=args.baseline_avg_num,
                 sample_time_window=args.window,
                 pivot=args.job_duration_pivot,
                 start_index=args.start_index,
                 filter_kernel=job_filter,
                 node=args.node)
    print("Simulation ends...")
    print("--------------------------------------------------------------------------------------------------------")

    print(f"Elapse Time: {datetime.today() - prev_time}")


if __name__ == '__main__':
    main()
