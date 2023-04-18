## Simple Slurm Simulator

***

### Timeline

1. Test Plan
    1. Basic tests for independent jobs
        - End, Update: 02/28/2022, End Date: 03/03/2022
    2. Basic tests for backfilling scheduler
        - End, Update: 03/02/2022, End Date: 03/08/2022
    3. Basic tests for simulator
        - End, Update: 03/08/2022, End Date: 03/09/2022
    4. Tests with jobs from longhorn/frontera within 48 hours
        - End, Update: 03/08/2022, End Date: 03/09/2022
    5. Tests with jobs from longhorn/frontera
        - Pending, Plan to start on 03/11/2022

2. Feature Development
    1. Simulate with given cluster state and new jobs
        - End, Update: 03/06/2022, End Date 03/08/2022
    2. Checkpoint
        - Pending
    3. Code reconstruction
       - Pending

### Manual

The detail is in simulator_test.py.

The following instruction is based on the test_longhorn_two_days().

1. Set config and initial jobs
   ```python
   job_log = "./test/filtered-longhorn-v100.log"
   slurm_config = "./test/slurm_config.json"
   backfill_config = "./test/backfill_config.json"
   ```
    * The job logs can be changed to any initial logs from longhorn and frontera but the format should be the same.
    * The config files should be json files includes all the key and values in the example file

2. Load configs
    ```python
    from parser_log import *
    slurm_config = load_slurm_config(slurm_config)
    backfill_config = load_backfill_config(backfill_config)
    ```

3. Initialize the simulator
   ```python
   simulator_tmp = simulator.Simulator()
   # load_jobs is optional and it depends on whether you want to use the job from the initial logs
   simulator_tmp.load_jobs(job_log)
   # Time should be datetime.datetime
   start_time = datetime.strptime("2020-06-26T08:40:53", "%Y-%m-%dT%H:%M:%S")
   # Every new simulator instance must initialize the scheduler
   simulator_tmp.init_scheduler(88, slurm_config, start_time, backfill_config)
   ```
    * The arguments of init_scheduler are number of nodes, slurm config, the start time of the simulator, and
      backfilling config.
    * It only supports backfilling currently.

4. Submit the jobs
    ```python
    # find_jobs can be used to search all the jobs within specific time and automatically add them into the execution job list of simulator
    simulator_tmp.find_jobs(datetime.strptime("2020-06-26T08:44:53", "%Y-%m-%dT%H:%M:%S"), datetime.strptime("2020-06-29T08:44:53", "%Y-%m-%dT%H:%M:%S"))
    # Self-defined job
    job_1 = job.Job("222", 87, "2020-06-26T08:44:53", "2020-06-26T08:44:50", "2020-06-28T08:44:53", 2880, 6000)
    # Submit self-defined job to the simulator
    simulator_tmp.submit_job_external([job_1])
    ```
    ```python
    # If you want to submit the jobs in the log, you can use job id
    simulator_tmp.submit_job_internal(["2", "3", "4", "5"])
    ```

5. Run simulation
    ```python
    simulator_tmp.run_end()
    logs = sorted(simulator_tmp.job_completion_logs, key=lambda x: x.start)
    ```
    * The logs will be in simulator_tmp.job_completion_logs.
    * Currently, running simulator for certain time is not supported (If you make sure there will be no job submitted
      during time you set, you can use run_time()).

6. Reconfig the simulation state
    ```python
    job_1 = job.Job("222", 87, "2019-11-04T17:20:00", "2019-11-04T17:20:00", "2019-11-06T17:20:00", 2880, 6000)
    job_2 = job.Job("333", 4, "2019-11-06T17:20:00", "2019-11-04T17:21:00", "2019-11-08T17:20:00", 2880, 6000)
    new_job = job.Job("444", 1, "2019-11-04T17:28:00", "2019-11-04T17:23:00", "2019-11-06T17:28:00", 2880, 6000)
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
    new_state = StateInput(curr_time, [job_1_log, job_2_log])
    ```
   * For the jobs that are running, start time, end time, and status need to be set as what job 1 looks like. The running time and node usage should also be set.
   * For the jobs that are pending in the queue, only status need to be set as PENDING.

7. Run simulation for reconfiguration and one self-define job
    ```python
    # If you want to check the actual start time of the new log, you need to provide start time and end time for the jobs after this self-defined job
    new_job_log = simulator_tmp.reconfig_exec(new_state, new_job)
    ```
    * The start time and end time of the jobs after the self-defined jobs can be set and the default value is None.
    * This method will only return the start time of the new job and will let the simulator stops at the time this self-define job is finished.