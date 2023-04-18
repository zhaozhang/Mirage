import sys
from typing import List, Tuple

sys.path.insert(2, 'src/simulator')

import job
import simulator
from simparser import SimParser
from datetime import datetime
from datetime import timedelta
import random
import logging


class Env:
    """
        Init the simulator and the second dependent job
    """

    def __init__(self, job_log, slurm_config, backfill_config,
                 sim_start_time: datetime.strptime, log_start_time: datetime.strptime, log_end_time: datetime.strptime):
        self.slurm_config = SimParser.load_slurm_config(slurm_config)
        self.backfill_config = SimParser.load_backfill_config(backfill_config)
        self.sim = simulator.Simulator()
        self.sim.load_jobs(job_log)
        self.sim_start_time = sim_start_time
        self.log_start_time = log_start_time
        self.log_end_time = log_end_time
        self.flag = False
        self.total_interrupt = 0
        self.job_2 = job.Job("333", 1, None, log_end_time, None, 2880,
                             6000, False)

    '''
        This function processes the machine state and returns a Nx3 feature matrix
    '''

    def state(self) -> List[Tuple]:  # returns sequence length x 3
        # running_state <- job_info keyed by job_id and value is job_log
        # pending_state <- list of jobs in scheduler _pending_queuue
        state_rep = self.sim.output_state()
        output = []
        running_jobs = state_rep.running_jobs
        pending_jobs = state_rep.pending_jobs
        for _, job_log in running_jobs.items():
            output.append(
                ((job_log.start - job_log.job.submit).total_seconds(), job_log.job.time_limit, job_log.job.nodes))
        for job in pending_jobs:
            output.append((-1, job.time_limit, job.nodes))
        return output

    def job_finish(self) -> bool:
        # query the state of the first job
        # returns job state
        state_rep = self.sim.output_state()
        running_jobs = state_rep.running_jobs
        pending_jobs = state_rep.pending_jobs
        # print(f"running_jobs: {running_jobs.keys()}")
        # print(f"pending_jobs: {[j.job_id for j in pending_jobs]}")

        if self.job_1.job_id in running_jobs:
            print(f"job {self.job_1.job_id} is running", flush=True)
            self.flag = True
            return False
        elif self.job_1.job_id in [j.job_id for j in pending_jobs]:
            print(f"job {self.job_1.job_id} is pending", flush=True)
            self.flag = False
            return False
        else:
            if self.flag:
                print(f"job {self.job_1.job_id} is done", flush=True)
                return True
            else:
                print(f"job {self.job_1.job_id} is submitted", flush=True)
                return False

    def job_finish_before_avg(self) -> bool:
        # query the state of the first job
        # returns job state
        state_rep = self.sim.output_state()
        running_jobs = state_rep.running_jobs
        pending_jobs = state_rep.pending_jobs
        job_logs = self.sim.job_completion_logs
        cur_time = self.sim.sim_time

        # print(f"running_jobs: {running_jobs.keys()}")
        # print(f"pending_jobs: {[j.job_id for j in pending_jobs]}")
        # print(f"job_logs: {[j.job.wait_time.total_seconds() for j in job_logs]}")
        # print(f"cur_time: {cur_time}")

        wait_avg = 0.0
        if len(job_logs) > 0:
            time_list = [int(j.job.wait_time.total_seconds() / 60.0) for j in job_logs]
            wait_avg = sum(time_list) / len(time_list)
            # print(f"avg_queue: {wait_avg}")

        if self.job_1.job_id in running_jobs:
            # print(f"job {self.job_1.job_id} is running", flush=True)
            self.flag = True
            end_time = running_jobs[self.job_1.job_id].end
            diff_time = (end_time - cur_time).total_seconds()
            print(f"diff: {diff_time}, avg: {wait_avg}", flush=True)

            if diff_time <= wait_avg:
                # print(f"time to submit job_2", flush=True)
                return True
            else:
                return False
        elif self.job_1.job_id in [j.job_id for j in pending_jobs]:
            print(f"job {self.job_1.job_id} is pending", flush=True)
            self.flag = False
            return False
        else:
            if self.flag:
                print(f"job {self.job_1.job_id} is done", flush=True)
                return True
            else:
                print(f"job {self.job_1.job_id} is submitted", flush=True)
                return False

    ''' 
        Step function applies an action (either submit/noOp)
        Then it runs for 1 minute
        Then computes rewards generated

    '''

    def step(self, action):
        done = False
        if action:
            self.job_2.submit = self.cur_time
            self.sim.submit_job_external([self.job_2])
            logging.info("Submitted job 2 at {}".format(self.job_2.submit))
            done = True
        elif self.total_interrupt < -1000:
            done = True
        self.sim.run_time(timedelta(minutes=1))
        self.cur_time = self.cur_time + timedelta(minutes=1)
        next_state = self.sim.output_state()
        if not done:
            reward = self.reward(next_state)
            logging.info("Reward:{}".format(reward))
        else:
            reward = self.final_reward(next_state)
            logging.info("Reward:{}".format(reward))

        return self.state(), reward, done

    '''
        Resets the machine and randomly runs the machine from sim_start_time to some random 
        time in the interval (sim_start_time, log_end_time)

        update: reset now chooses random time between sim_start_time and log_end_time and submits 
        a day's worth of jobs from this time. also sets simulator start time to chosen starttime
         and submits job1. This version does not simulate jobs submitted before job1

    '''

    def reset(self):
        logging.info("resetting")
        self.flag = False
        self.total_interrupt = 0
        start_time = self.sim_start_time
        end_time = self.log_end_time

        init_queue = random.random()
        print(f"init_queue: {init_queue}", flush=True)
        init_queue_time = timedelta(seconds=int(init_queue * ((end_time - start_time).total_seconds())))
        init_queue_time = init_queue_time + start_time
        print(f"init_queue_time: {init_queue_time}", flush=True)
        three_days = init_queue_time + timedelta(days=3)

        self.sim.reset_submitted_jobs()
        self.sim.init_scheduler(88, self.slurm_config, init_queue_time, self.backfill_config)
        jobs_log = self.sim.find_jobs(init_queue_time, three_days)
        self.sim.submit_job_internal(jobs_log)

        # init_queue_time = start_time + timedelta(minutes=2)# for debugging
        init_queue_time.replace(microsecond=0)

        # self.sim.run_time(init_queue_time - start_time)
        # self.sim.run_time(timedelta(seconds=0))
        logging.info("Submitted job 1 at {}".format(init_queue_time))
        self.job_1 = job.Job("222", 1, None, init_queue_time, None, 2880, 6000, False)
        self.sim.submit_job_external([self.job_1])
        self.cur_time = init_queue_time

    '''
        Computes the reward when NoOp. Only interrupt is possible here
    '''

    def reward(self, next_state: simulator.StateOutput):
        # if job1 still in queue or still running, reward is zero
        # if job1 has completed, then reward is -interrupt within last timestep
        reward = 0
        completed = self.sim.job_completion_logs
        for job_log in completed:
            if job_log.job.job_id == self.job_1.job_id:
                end = job_log.end
                time_diff = next_state.time - end
                if time_diff > timedelta(minutes=1):
                    reward = -1
                else:
                    reward = time_diff.total_seconds() / 60.0
        self.total_interrupt += reward
        return reward

    '''
        Runs the machien till the end. Both interrupt and overlap possible here
        Interrupt calculation should be adjusted because right now it is double rewarding 
        (giving reward for states that were rewarded)

        Update: now final_reward does not reward interrupt since interrupt signal
        is rewarded during each step

    '''

    def final_reward(self,
                     next_state: simulator.StateOutput):  # reward when you know job2 is submitted and we are terminating
        logging.info("final_reward func called at time : {}".format(next_state.time))
        # rest_jobs = self.sim.find_jobs(next_state.time, self.log_end_time)
        # self.sim.submit_job_internal(rest_jobs)
        if self.total_interrupt < -1000: return 2 * self.total_interrupt
        # self.sim.run_time(self.log_end_time+timedelta(days=2)-next_state.time)
        self.sim.run_end(self.job_2.job_id)
        logging.info("finished running to end: {}".format(self.sim.output_state().time))

        queried = 0
        job1_log = None
        job2_log = None

        # print(f"running_jobs: {running_jobs.keys()}")
        # print(f"pending_jobs: {[j.job_id for j in pending_jobs]}")

        completed = self.sim.job_completion_logs

        for job_log in completed:
            if queried == 2: break
            if job_log.job.job_id == self.job_1.job_id:
                job1_log = job_log
                queried += 1
            elif job_log.job.job_id == self.job_2.job_id:
                job2_log = job_log
                queried += 1
        interrupt = 0
        overlap = 0
        print("Job 1 log end time: {}".format(job1_log.end))
        print("Job 2 log start time: {}".format(job2_log.start))
        if job1_log.end < job2_log.start:
            logging.info("There is interrupt")
            interrupt = (job2_log.start - job1_log.end).total_seconds()
            interrupt /= -60.0
        if job1_log.end is not None and job1_log.end > job2_log.start:
            logging.info("There is overlap")
            overlap = ((job1_log.end - job2_log.start)).total_seconds()
            overlap /= -60.0
        print(f"final_rewards: interrupt: {interrupt}, overlap: {overlap}")
        return interrupt + 0.5 * overlap


def test_env():
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    job_log = "/scratch1/08105/shrey_k/interrupt-free-provisioning/src/simulator/test/filtered-longhorn-v100.log"
    slurm_config = "/scratch1/08105/shrey_k/interrupt-free-provisioning/src/simulator/test/slurm_config.json"
    backfill_config = "/scratch1/08105/shrey_k/interrupt-free-provisioning/src/simulator/test/backfill_config.json"

    start_time = datetime.strptime("2020-06-26T08:43:53", "%Y-%m-%dT%H:%M:%S")
    log_start_time = datetime.strptime("2020-06-26T08:44:53", "%Y-%m-%dT%H:%M:%S")
    log_end_time = datetime.strptime("2020-06-26T09:03:00", "%Y-%m-%dT%H:%M:%S")

    env = Env(job_log, slurm_config, backfill_config, start_time, log_start_time, log_end_time)

    for ep in range(10):
        env.reset()
        count = 0
        print("episode count : {}".format(ep))
        while count < 1000:
            count += 1
            if count % 5 == 0:
                _, r, done = env.step(1)
            else:
                _, r, done = env.step(0)
            print(r)
            if done:
                break


if __name__ == '__main__':
    test_env()
