import random
import unittest
import simulator
from parser_log import *
import job
from simulator import StateInput
from datetime import timedelta
import logging
import pickle


class SimulatorTestCase(unittest.TestCase):
    def test_basic(self):
        job_log = "./test_data/filtered-longhorn-v100.log"
        slurm_config = "./test_data/slurm_config.json"
        backfill_config = "./test_data/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)
        simulator_tmp = simulator.Simulator()
        simulator_tmp.load_jobs(job_log)
        start_time = datetime.strptime("2019-11-04T17:20:00", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)
        simulator_tmp.submit_job_internal(["2", "3", "4", "5"])
        simulator_tmp.run_end()
        print("------------------------------------------Job Logs-----------------------------------------------------")
        for item in simulator_tmp.job_completion_logs:
            print("Job ID: {0}, Job Submit: {1}, Job Start: {2}, Job End: {3}, Job Nodes: {4}".format(item.job.job_id,
                                                                                                      item.job.submit,
                                                                                                      item.start,
                                                                                                      item.end,
                                                                                                      item.nodes))
        print("-------------------------------------------------------------------------------------------------------")

    def test_basic_backfill(self):
        job_1 = job.Job("222", 87, "2019-11-04T17:20:00", "2019-11-04T17:20:00", "2019-11-06T17:20:00", 2880, 6000,
                        False)
        job_2 = job.Job("333", 4, "2019-11-06T17:20:00", "2019-11-04T17:21:00", "2019-11-08T17:20:00", 2880, 6000,
                        False)
        job_3 = job.Job("444", 1, "2019-11-04T17:22:00", "2019-11-04T17:22:00", "2019-11-06T17:22:00", 2880, 6000,
                        False)
        slurm_config = "./test_data/slurm_config.json"
        backfill_config = "./test_data/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)
        simulator_tmp = simulator.Simulator()
        start_time = datetime.strptime("2019-11-04T17:20:00", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)
        simulator_tmp.submit_job_external([job_1, job_2, job_3])
        simulator_tmp.run_end()
        print("------------------------------------------Job Logs-----------------------------------------------------")
        for item in simulator_tmp.job_completion_logs:
            print("Job ID: {0}, Job Submit: {1}, Job Start: {2}, Job End: {3}, Job Nodes: {4}".format(item.job.job_id,
                                                                                                      item.job.submit,
                                                                                                      item.start,
                                                                                                      item.end,
                                                                                                      item.nodes))
        print("-------------------------------------------------------------------------------------------------------")

    def test_basic_reconfig(self):
        logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            level=logging.INFO)
        job_1 = job.Job("222", 87, "2019-11-04T17:20:00", "2019-11-04T17:20:00", "2019-11-06T17:20:00", 2880, 6000,
                        False)
        job_2 = job.Job("333", 4, "2019-11-06T17:20:00", "2019-11-04T17:21:00", "2019-11-08T17:20:00", 2880, 6000,
                        False)
        new_job = job.Job("444", 1, "2019-11-04T17:28:00", "2019-11-04T17:23:00", "2019-11-06T17:28:00", 2880, 6000,
                          False)
        curr_time = datetime.strptime("2019-11-04T17:22:00", "%Y-%m-%dT%H:%M:%S")
        job_1_log = job.JobLog(job_1)
        for i in range(0, 87):
            job_1_log.add_nodes(i)
        job_1_log.start = datetime.strptime("2019-11-04T17:20:00", "%Y-%m-%dT%H:%M:%S")
        job_1_log.status = job.JobStatus.RUNNING
        job_1_log.run(curr_time - job_1_log.start)
        job_1_log.end = job_1_log.start + job_1.duration

        job_2_log = job.JobLog(job_2)
        job_2_log.status = job.JobStatus.PENDING

        slurm_config = "./test_data/slurm_config.json"
        backfill_config = "./test_data/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)
        simulator_tmp = simulator.Simulator()
        simulator_tmp.init_scheduler(88, slurm_config, curr_time, backfill_config)

        new_state = StateInput(curr_time, [job_1_log, job_2_log])
        simulator_stop_time = simulator_tmp.reconfig_exec(new_state, new_job)
        print(simulator_stop_time)

    def test_longhorn_two_days(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
        job_log = "./test_data/filtered-longhorn-v100.log"
        slurm_config = "./test_data/slurm_config.json"
        backfill_config = "./test_data/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)

        simulator_tmp = simulator.Simulator()
        simulator_tmp.load_jobs(job_log)

        start_time = datetime.strptime("2020-06-26T08:40:53", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)

        jobs_log = simulator_tmp.find_jobs(datetime.strptime("2020-06-26T08:44:53", "%Y-%m-%dT%H:%M:%S"),
                                           datetime.strptime("2020-06-29T08:44:53", "%Y-%m-%dT%H:%M:%S"))
        simulator_tmp.submit_job_internal(jobs_log)

        # Priority initialization
        job_1 = job.Job("222", 86, "2020-06-24T08:44:53", "2020-06-26T08:44:50", "2020-06-26T08:44:53", 2880, 6000,
                        False)
        simulator_tmp.submit_job_external([job_1])

        simulator_tmp.run_end()

        logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
        with open("./test_out/test_1.log", "w") as f:
            f.write(
                "------------------------------------------Job Logs-------------------------------------------------\n")
            for item in logs:
                f.write(item.info_str() + "\n")
            f.write(
                "---------------------------------------------------------------------------------------------------")

    def test_run_time(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
        job_log = "./test_data/filtered-longhorn-v100.log"
        slurm_config = "./test_data/slurm_config.json"
        backfill_config = "./test_data/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)

        simulator_tmp = simulator.Simulator()
        simulator_tmp.load_jobs(job_log)

        start_time = datetime.strptime("2020-06-26T08:40:53", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)

        jobs_log = simulator_tmp.find_jobs(datetime.strptime("2020-06-26T08:44:53", "%Y-%m-%dT%H:%M:%S"),
                                           datetime.strptime("2020-06-29T08:44:53", "%Y-%m-%dT%H:%M:%S"))
        simulator_tmp.submit_job_internal(jobs_log)

        # Priority initialization
        job_1 = job.Job("222", 86, "2020-06-24T08:44:53", "2020-06-26T08:44:50", "2020-06-26T08:44:53", 2880, 6000,
                        False)
        simulator_tmp.submit_job_external([job_1])
        simulator_tmp.run_time(
            datetime.strptime("2020-07-02T09:13:55", "%Y-%m-%dT%H:%M:%S") - datetime.strptime("2020-06-26T08:40:53",
                                                                                              "%Y-%m-%dT%H:%M:%S"))

        logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
        with open("test_out/test_2.log", "w") as f:
            f.write(
                "------------------------------------------Job Logs-------------------------------------------------\n")
            for item in logs:
                f.write(item.info_str() + "\n")
            f.write(
                "---------------------------------------------------------------------------------------------------")

    def test_submit_job_dependency(self):
        job_log = "./test_data/filtered-longhorn-v100.log"
        slurm_config = "./test_data/slurm_config.json"
        backfill_config = "./test_data/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)

        simulator_tmp = simulator.Simulator()
        simulator_tmp.load_jobs(job_log)

        start_time = datetime.strptime("2020-06-26T08:40:53", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)

        jobs_log = simulator_tmp.find_jobs(datetime.strptime("2020-06-26T08:44:53", "%Y-%m-%dT%H:%M:%S"),
                                           datetime.strptime("2020-06-29T08:44:53", "%Y-%m-%dT%H:%M:%S"))
        simulator_tmp.submit_job_internal(jobs_log)

        job_1 = job.Job("222", 86, None, None, None, 2880, 6000, False)
        job_2 = job.Job("333", 1, None, None, None, 2880, 6000, False)
        simulator_tmp.submit_job_external([])

        def count():
            num = random.randint(1, 5000)
            if num > 4950:
                return True
            else:
                return False

        # The first job need to be same as the time of the simulator
        simulator_tmp.run_job_dependency(job_1, job_2, lambda: count())
        state = simulator_tmp.output_state()
        print(state.time)
        simulator_tmp.run_end()
        logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
        with open("./test_out/test_3.log", "w") as f:
            f.write(
                "------------------------------------------Job Logs-------------------------------------------------\n")
            for item in logs:
                f.write(item.info_str() + "\n")
            f.write(
                "---------------------------------------------------------------------------------------------------")

    def test_run_group_dependency(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
        job_log = "../test/filtered-longhorn-v100.log"
        slurm_config = "../test/slurm_config.json"
        backfill_config = "../test/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)

        simulator_tmp = simulator.Simulator(mode=simulator.Mode.MINUTES)
        simulator_tmp.load_jobs(job_log)

        start_time = datetime.strptime("2021-12-20T01:40:00", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)

        jobs_log = simulator_tmp.find_jobs(datetime.strptime("2020-06-01T08:44:00", "%Y-%m-%dT%H:%M:%S"),
                                           datetime.strptime("2020-07-01T08:44:00", "%Y-%m-%dT%H:%M:%S"))
        simulator_tmp.submit_job_internal(jobs_log)

        job_lst = []
        for i in range(0, 256, 2):
            time = datetime.strptime("2020-06-01T08:44:00", "%Y-%m-%dT%H:%M:%S") + timedelta(
                minutes=random.randint(0, 36000))
            job_lst.append([job.Job(str(i + 1000000), 1, None, time, None, 2880, 6000, False),
                            job.Job(str(i + 1000001), 1, None, None, None, 2880, 6000, False)])

        def count(pre_job, sim_time):
            if timedelta(minutes=2640) <= sim_time - pre_job.submit:
                return True
            return False

        def report(mini_batch, t, cache):
            with open("../test/test2.pickle", "wb") as f:
                pickle.dump(mini_batch, f)
            debug_time = t.strftime("%Y-%m-%dT%H:%M:%S")
            return True

        cache_tmp = {}
        # The first job need to be same as the time of the simulator
        simulator_tmp.run_group_dependency(job_lst, infer_func=lambda x: count(x, simulator_tmp.sim_time),
                                           train_func=lambda x: report(x, simulator_tmp.sim_time, cache_tmp),
                                           sample_time_length=600, train_freq=10, infer_freq=5, max_store_reward=127,
                                           sample_time_step=144)
        logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
        print(simulator_tmp.reward())
        logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
        with open("../test/test_5_6.log", "w") as f:
            f.write(
                "------------------------------------------Job Logs-------------------------------------------------\n")
            for item in logs:
                f.write(item.info_str() + "\n")
            f.write(
                "---------------------------------------------------------------------------------------------------")

    def test_load_data(self):
        pickle_file = "../test/test2.pickle"
        with open(pickle_file, "rb") as f:
            content = pickle.load(f)
        debug = 1

    def test_load_data_2(self):
        pickle_file = "../test/fake_interruption_overlap_pred.pickle"
        with open(pickle_file, "rb") as f:
            content = pickle.load(f)
        debug = 1

    def test_load_new_jobs(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
        slurm_config = "../test/slurm_config.json"
        backfill_config = "../test/backfill_config.json"
        slurm_config = load_slurm_config(slurm_config)
        backfill_config = load_backfill_config(backfill_config)

        simulator_tmp = simulator.Simulator(mode=simulator.Mode.MINUTES)
        pickle_file = "../../data/philly_trace_workload.pickle"
        base_job_log = "../../data/filtered-frontera-rtx.log"
        simulator_tmp.load_pickle_jobs(base_job_log, pickle_file)

        start_time = datetime.strptime("2017-08-14T23:26:00", "%Y-%m-%dT%H:%M:%S")
        simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)

        simulator_tmp.run_end()

        logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
        with open("../test/test_new_trace.log", "w") as f:
            f.write(
                "------------------------------------------Job Logs-------------------------------------------------\n")
            for item in logs:
                f.write(item.info_str() + "\n")
            f.write(
                "---------------------------------------------------------------------------------------------------")

    def test_load_pickle_jobs(self):
        with open("../../data/philly_trace_workload.pickle", "rb") as f:
            content = pickle.load(f)
        t = 0


if __name__ == '__main__':
    unittest.main()
