import random
from sim import *


class Env:
    """
        Init the simulator and the second dependent job
    """

    def __init__(self, job_log, slurm_config, backfill_config, sim_start_time, log_start_time, log_end_time, warmup_len,
                 sim_len, sim_window, seed, job_node):
        self.slurm_config = SimParser.load_slurm_config(slurm_config)
        self.backfill_config = SimParser.load_backfill_config(backfill_config)
        self.sim = simulator.Simulator(mode=Mode.MINUTES)
        self.sim.load_jobs(job_log)
        self.sim_start_time = datetime.strptime(sim_start_time, "%Y-%m-%dT%H:%M:%S")
        self.log_start_time = datetime.strptime(log_start_time, "%Y-%m-%dT%H:%M:%S")
        self.log_end_time = datetime.strptime(log_end_time, "%Y-%m-%dT%H:%M:%S")
        self.warmup_len = warmup_len
        self.sim_len = sim_len
        self.sim_window = sim_window
        self.random_seed = seed
        self.job_node = job_node
        self._random_set = False

    '''
        Resets the machine and randomly runs the machine from sim_start_time to some random 
        time in the interval (sim_start_time, log_end_time)

        update: reset now chooses random time between sim_start_time and log_end_time and submits 
        a day's worth of jobs from this time. also sets simulator start time to chosen starttime
         and submits job1. This version does not simulate jobs submitted before job1

    '''

    def reset(self):
        logging.info("Resetting")
        start_time = self.log_start_time
        end_time = self.log_end_time

        if not self._random_set:
            random.seed(self.random_seed)
            self._random_set = True
        time_offset = random.random()
        logging.info(f"Time offset: {time_offset}")
        sim_start_time = timedelta(minutes=int(time_offset * ((end_time - start_time).total_seconds()) / 60))
        sim_start_time = sim_start_time + start_time
        logging.info(f"Simulation Start Time: {sim_start_time}")
        sim_end_time = sim_start_time + timedelta(days=self.sim_len)

        self.sim.reset()
        self.sim.init_scheduler(88, self.slurm_config, sim_start_time, self.backfill_config)
        jobs_log = self.sim.find_jobs(sim_start_time, sim_end_time)
        self.sim.submit_job_internal(jobs_log)
        self.sim_start_time = sim_start_time

    def reward(self, infer_func):
        time = self.sim_start_time + timedelta(days=self.warmup_len)
        job_1 = job.Job("1000000", self.job_node, None, time, None, 2880, 6000, False)
        job_2 = job.Job("1000001", self.job_node, None, time, None, 2880, 6000, False)

        self.sim.run_group_dependency([[job_1, job_2]],
                                      infer_func=infer_func,
                                      train_func=None,
                                      sample_time_length=600,
                                      infer_freq=60,
                                      sample_time_step=self.sim_window,
                                      infer_lower_bound=0.5)
        overlap_interrupt = self.sim.reward()[0][3]
        # if overlap_interrupt <= 0.0:
        #    reward = 200 - abs(0.5 * 60 * overlap_interrupt)
        # else:
        #    reward = 60 - abs(60 * overlap_interrupt)
        reward = - 60 * abs(overlap_interrupt)
        return reward, overlap_interrupt
