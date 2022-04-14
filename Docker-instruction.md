## Pro2 Source Codes and User Manual

***

### File Directory

* /pro_2/train_model: Model train
* /pro_2/offline_data_gen: Training data generation and baseline generation
* /pro_2/online_validation: Online validation of the model

### User Manual

#### Start singularity and download the data seperately

1. This code was only tested on TACC Frontera supercomputer, plesae login the machine and start an idev session by running
   ```shell
      idev -N 1 -n 1 -p rtx-dev -t 02:00:00
   ```

2. After logging in, download the docker image
   ```shell
      singularity pull docker://ghcr.io/zhaozhang/pro_2:master
      singularity exec pro_2_master.sif bash
   ```

3. Copy the code to /dev/shm for write permission
   ```shell
      Singularity> cp -r /pro_2 /dev/shm/
      Singularity> cd /dev/shm/pro_2
   ```

4. The raw job trace data is proprietary, please contact zzhang@tacc.utexas.edu to get a copy of the job trace data, which is not publicly available. With the link, e.g., https://utexas.box.com/shared/static/hashvalue, run
   ```shell
      Singularity> curl -L https://utexas.box.com/shared/static/hashvalue --output offline_data_gen/filtered-longhorn-v100.log
   ```


#### Training Data Generation (offline_data_gen)

   ```shell
   # Training data will be generated as batch_2800_7.pickle
   Singularity> unset PYTHONPATH
   Singularity> cd offline_data_gen
   Singularity> python3 offline_data_gen.py -parallel -num_samples 2802 -num_probe 7 -interval 4
   ```

#### Baseline generation (offline_data_gen)

   Generating avg baseline and reactive baseline:
   ```shell
   # Pickle file structure:
   # All the rewards are in the first element for each list [[XXX, reward], ..., [XXX, reward]]
   
   # For average baseline
   Singularity> python3 offline_data_gen.py -parallel -baseline -num_samples 888 -interval 4
   
   # For reactive baseline
   # Choose the fifth sample per five samples and it will be the reactive baseline
   Singularity> python3 offline_data_gen.py -parallel -num_samples 888 -interval 4
   ```

#### Model train (train_model)
   You can use the previously generated training data or download from the provided link.
   ```shell
   # Create two file folders called "data" and "model" under current directory.
   Singularity> cd /dev/shm/pro_2
   Singularity> mdkir data model

   # Copy the training data into "data" folder.
   Singularity> cp offline_data_gen/batch_2800_7.pickle data/

   # You may also download the training data from box. The data is valid till 05/14/2022
   Singularity> curl -L https://utexas.box.com/shared/static/2dnwd4v1uvp59spqza14x3xosoncvuki --output data/batch_2800_7.pickle

   # Go to train_model/
   Singularity> cd train_model/

   # Train the model, the model will be model/model_2800_7.pt
   Singularity> python3 train.py
   ````

#### Online validation (online_validation)

   You can use the previously trained model or download the model from teh provided link

1. Copy the log trace so that Ray can run in parallel
   ```shell
   Singularity> cd /dev/shm/pro_2/online_validation
   Singularity> cp ../offline_data_gen/filtered-longhorn-v100.log .
   ```

2. Run
   ```shell
   # The output file will be reward.log under the same path
   Singularity> python3 top_validate.py -num_validate 888
   ```

3. You will see a reward.log that contains the validation rewards.
   

