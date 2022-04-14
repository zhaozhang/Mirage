## Pro2 Source Codes and User Manual

***

### System Requirement

The experiment is run on the following platform and dependencies:

* Python 3.9.2
* Ray 1.11.0
* tqdm 4.63.1
* torch 1.11.0
* cuda 11.3

### File Directory

* train_model: Model train
* offline_data_gen: Training data generation and baseline generation
* online_validation: Online validation of the model

### User Manual

#### Training Data Generation (offline_data_gen)

1. Workload Initialization
    ```python
    job_log = "./filtered-frontera-rtx.log"
    ```
    * Search the code above in offline_data_gen.py and replace it with the name of the workload.
    * This log file should be in the same path with all the scripts.
   ```python
   simulator_init_time = datetime.strptime("2020-11-14T00:00:00", "%Y-%m-%dT%H:%M:%S")
   ```
    * Search the code above and change it into which date you want as the start time of the workload.
    * Training Data in the experiment is from 2020-11-14T00:00:00 (start time of the workload) to 2021-02-27T00:00:00
2. Run
   ```shell
   # Training data will be generated as test.pickle
   python3 offline_data_gen.py -parallel -num_samples 2802 -num_probe 7 -interval 4
   ```

#### Baseline generation (offline_data_gen)

1. Workload Initialization
    ```python
    job_log = "./filtered-frontera-rtx.log"
    ```
    * Search the code above in offline_data_gen.py and replace it with the name of the workload.
    * This log file should be in the same path with all the scripts.
   ```python
   simulator_init_time = datetime.strptime("2020-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
   ```
    * Search the code above and change it into which date you want as the start time of the workload.
    * Training Data in the experiment is from 2020-03-01T00:00:00 (start time of the workload) to 2021-07-27T00:00:00
2. Run
   ```shell
   # Pickle file structure:
   # All the rewards are in the first element for each list [[XXX, reward], ..., [XXX, reward]]
   
   # For average baseline
   python3 offline_data_gen.py -parallel -baseline -num_samples 888 -interval 4
   
   # For reactive baseline
   # Choose the fifth sample per five samples and it will be the reactive baseline
   python3 offline_data_gen.py -parallel -num_samples 888 -interval 4
   ```

#### Model train (train_model)
1. Directory initialization
   * Create two file folders called "data" and "model" under current directory.
   * Copy the training data into "data" folder.
2. Model initialization
   ```python
   MODEL_NAME = 'transformer'
   PARALLEL_PREPROCESSING = True
   TRAIN_FRACTION = 2 / 3
   TRAIN_NEPOCHS = 200
   ```
   * MODEL_NAME can be changed into "transformer", "convolution", or "linear".
   * PARALLEL_PREPROCESSING can be enabled when ray is installed.
   * TRAIN_NEPOCHS is the epoch of training.
3. Reward initialization
   ```python
   fname_data_raw = 'batch_2802_7_node_8.pickle'
   fname_data_tensor = 'batch_2802_7_node_8_cache.pickle'
   fname_model = 'model_2802_7_node_8_convolution.pt'
   ```
    * Search codes above in train.py and change the first line (fname_data_raw) into the name of training data file.
    * The second line and third line are output files, and it can be set as any name you want.
    * The output will be in "./model".

4. Run
   ```shell
   python3 train.py
   ```

#### Online validation (online_validation)

1. Model initialization
   ```python
   checkpoint_path = "../model_2800_7_convolution.pt"
   ```
    * Search the code above in top_validate.py and change it into the location of your model file.
    * This model file can be anywhere, and it is recommended to put it outside the directory of source codes.

2. Workload initialization
   ```python
   start_time = datetime.strptime("2021-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S") + timedelta(hours=loop * args.interval)
   ```
    * "2021-03-01T00:00:00" can be changed into any start date of the workload and in this experiment, it is "
      2021-03-01T00:00:00".

4. Run
   ```shell
   # The output file will be reward.log under the same path
   python3 top_validate.py -num_validate 888
   ```
   

