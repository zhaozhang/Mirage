import pickle
import torch
import ray
import argparse
import os
import sys
from torch.distributions import Categorical

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath("../model/"))
import sim
from sim import *
from moe import *
from quantile import quantile
from moe.model import TransformerRegressor


@ray.remote
def simulate(interval, loop, infer_wrapper, slurm_cfg, backfill_cfg, workload, start_time, warmup_len, workload_len,
             regr_model, sample_window, infer_low_bound, num_node=1):
    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if regr_model is None:
        jobs = SimParser.parse_job(workload)
    else:
        jobs = SimParser.parse_pickle_job(workload, regr_model.predict)
    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs
    # ---------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------------
    # Inference procedure
    start_time = start_time + timedelta(hours=loop * interval)
    logging.info(f"Loop: {loop}, Start Time: {start_time}")
    workload_start_time = start_time
    workload_end_time = workload_start_time + timedelta(days=workload_len)
    slurm_simulator.reset_history()
    slurm_simulator.reset_submitted_jobs()
    slurm_simulator.init_scheduler(88, slurm_cfg, start_time, backfill_cfg)
    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)

    time = workload_start_time + timedelta(days=warmup_len)
    job_1 = job.Job("1000000", num_node, None, time, None, 2880, 6000, False)
    job_2 = job.Job("1000001", num_node, None, time, None, 2880, 6000, False)

    slurm_simulator.run_group_dependency([[job_1, job_2]],
                                         infer_func=infer_wrapper,
                                         train_func=None,
                                         sample_time_length=600,
                                         infer_freq=10,
                                         sample_time_step=sample_window,
                                         infer_lower_bound=infer_low_bound)
    return slurm_simulator.reward(), slurm_simulator.prediction_trace, loop


def simulate_serial(interval, loop, infer_wrapper, slurm_cfg, backfill_cfg, workload, start_time, warmup_len,
                    workload_len, regr_model, sample_window):
    # ---------------------------------------------------------------------------------------------------------------
    # Simulator Initialization
    # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if regr_model is None:
        jobs = SimParser.parse_job(workload)
    else:
        jobs = SimParser.parse_pickle_job(workload, regr_model.predict)
    slurm_simulator = Simulator(mode=Mode.MINUTES)
    slurm_simulator.jobs = jobs
    # ---------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------------
    # Inference procedure
    start_time = start_time + timedelta(hours=loop * interval)
    logging.info(f"Loop: {loop}, Start Time: {start_time}")
    workload_start_time = start_time
    workload_end_time = workload_start_time + timedelta(days=workload_len)
    slurm_simulator.reset_history()
    slurm_simulator.reset_submitted_jobs()
    slurm_simulator.init_scheduler(88, slurm_cfg, start_time, backfill_cfg)
    jobs_log = slurm_simulator.find_jobs(workload_start_time, workload_end_time)
    slurm_simulator.submit_job_internal(jobs_log)

    time = workload_start_time + timedelta(days=warmup_len)
    job_1 = job.Job("1000000", 1, None, time, None, 2880, 6000, False)
    job_2 = job.Job("1000001", 1, None, time, None, 2880, 6000, False)

    infer_bound_pending_jobs, infer_bound_running_logs, infer_bound_nodes = slurm_simulator.run_group_dependency(
        [[job_1, job_2]],
        infer_func=infer_wrapper,
        train_func=None,
        sample_time_length=600,
        infer_freq=60,
        sample_time_step=sample_window,
        infer_lower_bound=-0.5)
    return slurm_simulator.reward(), slurm_simulator.prediction_trace, loop, infer_bound_pending_jobs, infer_bound_running_logs, infer_bound_nodes


def main():
    # ---------------------------------------------------------------------------------------------------------------
    # Argument Parse
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_validate", type=int, default=1)
    parser.add_argument("-interval", type=int, default=4)
    parser.add_argument("-workload", "--workload_name", type=str, default="filtered-longhorn-v100.log")
    parser.add_argument("-workload_len", type=int, default=3)
    parser.add_argument("-parallel", action="store_true", default=False)
    parser.add_argument("-start_time", type=lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"),
                        default="2021-03-01T00:00:00")
    parser.add_argument("-od", "--output_dir", default="./")
    parser.add_argument("-cd", "--config_dir", default="../test/test_data")
    parser.add_argument("-m", "--model", default="../model_2800_7_convolution.pt")
    parser.add_argument("-warmup_len", type=int, default=2)
    parser.add_argument("-philly", action="store_true", default=False)
    parser.add_argument("-philly_tl", "--philly_time_limit")
    parser.add_argument("-sample_window", type=int, default=144)
    parser.add_argument("-mt", "--model_type", type=str, default="moe")
    parser.add_argument("-node", type=int, default=1)
    args = parser.parse_args()

    prev_time = datetime.today()
    os.makedirs(args.output_dir, exist_ok=True)
    from moe import model

    model_pred = None
    infer_wrapper = None
    infer_low_bound = -0.5

    # ---------------------------------------------------------------------------------------------------------------
    # Simulation Initialization
    if args.model_type == "moe" or args.model_type == "policy-gradient":
        checkpoint = torch.load(args.model, map_location=torch.device("cpu"))

        model_name = checkpoint['model_name']
        model_hparams = checkpoint['model_hparams']
        model_state = checkpoint['model_state_dict']
        optimizer_name = checkpoint['optimizer_name']
        optimizer_state = checkpoint['optimizer_state_dict']

        model_pred = Predictor(model_name=model_name, hparams=model_hparams, optimizer_name=optimizer_name,
                               model_state=model_state, optimizer_state=optimizer_state,
                               base_learning_rate=1e-4, device='gpu')

        def infer_wrapper_moe(_, data_input):
            if data_input[0][0][2][3] == 172800.0:
                return 0.0

            np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False)

            # From numpy array to pytorch tensor
            tensor_input = torch.from_numpy(np_ndarray_input)
            # logging.debug(f"Complete inference for one job pair")
            return model_pred.inference(tensor_input).data[0].item()

        def infer_wrapper_policy_gradient(metadata, data_input):
            np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False)

            if data_input[0][0][2][3] == 172800.0:
                return 1.0
            # available_nodes = metadata[1]
            # if available_nodes >= 1:
            #     return -1.0

            # From numpy array to pytorch tensor
            tensor_input = torch.from_numpy(np_ndarray_input)
            tensor_output = model_pred.inference(tensor_input)
            future_action = Categorical(logits=tensor_output).sample()

            return future_action.item()

        if args.model_type == "moe":
            infer_wrapper = infer_wrapper_moe
        else:
            infer_wrapper = infer_wrapper_policy_gradient
            infer_low_bound = 0.5
    elif args.model_type == "baseline":
        print("Loading Baseline Regression model...")
        with open(args.model, "rb") as f:
            model_pred = pickle.load(f)

        def infer_wrapper_baseline(_, data_input):
            if data_input[0][0][2][3] == 172800.0:
                return 0.0

            np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False).reshape(1, -1)
            return model_pred.predict(np_ndarray_input)[0]

        infer_wrapper = infer_wrapper_baseline
    elif args.model_type == "quantile":
        quantile_pred = quantile.QuantileModel(0.99, 0.95)

        def infer_wrapper_quantile(metadata, data_input):
            if data_input[0][0][2][3] == 172800.0:
                return 0.0
            retired_jobs = metadata[0]
            if len(retired_jobs) == 0.0:
                return -1.0

            return quantile_pred.predict(retired_jobs)

        infer_wrapper = infer_wrapper_quantile
    else:
        print("Error model type")
        exit(1)

    slurm_config = SimParser.load_slurm_config(os.path.join(args.config_dir, "slurm_config.json"))
    backfill_config = SimParser.load_backfill_config(os.path.join(args.config_dir, "backfill_config.json"))

    print("--------------------------------------------------------------------------------------------------------")
    print("Simulation starts...")

    print("-----------------------")
    regr = None
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

    prediction_trace_data = {}
    total_reward = {}

    if args.parallel:
        rpc = []
        finish_count = 0
        total_finish_time = timedelta(seconds=0)
        file = os.listdir("../workload")
        file = list(filter(lambda x: x != args.workload_name, file))
        file = list(map(lambda x: "/workload/" + x, file))
        ray.init(runtime_env={"py_modules": [sim], "working_dir": "../",
                              "excludes": ["/test/", "/data/", "/misc/"] + file})

        for i in range(0, args.num_validate):
            print(f"Submitting job {i + 1}/{args.num_validate}...")
            rpc.append(
                simulate.remote(args.interval, i, infer_wrapper, slurm_config, backfill_config,
                                os.path.join("./workload", args.workload_name), args.start_time,
                                args.warmup_len, args.workload_len, regr, args.sample_window, infer_low_bound,
                                args.node))
        print("-----------------------")
        while len(rpc):
            wait_time = datetime.today()
            done_id, rpc = ray.wait(rpc)
            finish_count += 1
            res = ray.get(done_id[0])
            total_reward[res[2]] = res[0]
            prediction_trace_data[res[2]] = res[1]
            total_finish_time += datetime.today() - wait_time
            avg_complete_sec = total_finish_time.total_seconds() / finish_count
            expected_running_time = avg_complete_sec * (args.num_validate - finish_count)
            print(
                f"Job Finish {finish_count}/{args.num_validate}, "
                f"Average finish time: "
                f"{'{}:{}:{}'.format(avg_complete_sec // 3600, avg_complete_sec % 3600 // 60, avg_complete_sec % 60)}, "
                f"Expected running time: "
                f"{'{}:{}:{}'.format(expected_running_time // 3600, expected_running_time % 3600 // 60, expected_running_time % 60)}")
    else:
        for i in range(0, args.num_validate):
            print(f"Submitting job {i + 1}/{args.num_validate}...")
            reward, pred_trace, _, pend_jobs, run_logs, nodes = simulate_serial(args.interval,
                                                                                i,
                                                                                infer_wrapper,
                                                                                slurm_config,
                                                                                backfill_config,
                                                                                os.path.join("../workload",
                                                                                             args.workload_name),
                                                                                args.start_time,
                                                                                args.warmup_len,
                                                                                args.workload_len,
                                                                                regr,
                                                                                args.sample_window)
            total_reward[i] = reward
            prediction_trace_data[i] = pred_trace
            run_logs = sorted(run_logs, key=lambda x: x.job.submit)
            with open(os.path.join(args.output_dir, f"job_pending_{i}.log"), "w") as f:
                f.write(
                    "------------------------------------------Job Logs-------------------------------------------------\n")
                for item in pend_jobs:
                    f.write(item.info_str() + "\n")
                f.write(
                    "---------------------------------------------------------------------------------------------------\n")
                f.write(f"Available nodes: {nodes}\n")
            with open(os.path.join(args.output_dir, f"job_running_logs_{i}.log"), "w") as f:
                f.write(
                    "------------------------------------------Job Logs-------------------------------------------------\n")
                for item in run_logs:
                    f.write(item.info_str() + "\n")
                f.write(
                    "---------------------------------------------------------------------------------------------------\n")

    total_reward = dict(sorted(total_reward.items()))
    prediction_trace_data = dict(sorted(prediction_trace_data.items()))

    print("-----------------------")
    print("Simulation ends...")
    print("--------------------------------------------------------------------------------------------------------")
    with open(os.path.join(args.output_dir, "reward.pickle"), "wb") as f:
        pickle.dump(total_reward, f)
    with open(os.path.join(args.output_dir, "reward.log"), "w") as f:
        f.write(
            "------------------------------------------Reward Logs-------------------------------------------------\n")
        f.write("pred_job_id,succ_job_id,expected_reward,actual_reward,reward_gen_time,index,backfill,submission\n")
        for key, value in total_reward.items():
            if value is None:
                f.write("None\n")
            else:
                f.write(
                    f"{value[0][0]},{value[0][1]},{value[0][2]},{value[0][3]},{value[0][4]},{key},{value[0][5]},{value[0][6]}\n")
        f.write(
            "---------------------------------------------------------------------------------------------------")
    with open(os.path.join(args.output_dir, "prediction_trace.tickle"), "wb") as f:
        pickle.dump(prediction_trace_data, f)

    print(f"Elapse Time: {datetime.today() - prev_time}")


if __name__ == '__main__':
    main()
