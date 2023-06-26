## Pro2 Source Code and User Manual

***

### File Directory

* /Mirage/train_model: Model train
* /Mirage/offline_data_gen: Training data generation and baseline generation
* /Mirage/online_validation: Online validation of the model

### Data

There are three types of data that we use in our project:

* job traces: derived from scheduler logs. The raw job trace is proprietary; please contact
  zzhang@tacc.utexas.edu if you want to get access.
* processed training data:  this includes training data that is derived from the job traces and is
  the output offline_data_gen. We have made this available on Box until 5/14/2022.
* trained model: this is the output from training our model by running train_model. We have also made this available on Box.

### User Manual

#### Start singularity and download the data seperately

1. This code was only tested on TACC Frontera supercomputer, please login to the machine and start an idev session by running
   ```shell
      idev -N 1 -n 1 -p rtx-dev -t 02:00:00
   ```

2. After logging in, download the docker image
   ```shell
      singularity pull docker://ghcr.io/zhaozhang/Mirage:master
      singularity exec Mirage_master.sif bash
   ```

3. Copy the code to /dev/shm for write permission
   ```shell
      Singularity> cp -r /Mirage /dev/shm/
      Singularity> cd /dev/shm/Mirage
   ```

4. The raw job trace data is proprietary, please contact zzhang@tacc.utexas.edu to get a copy of the job trace data, which is not publicly available. With the link, e.g., https://utexas.box.com/shared/static/hashvalue, run
   ```shell
      Singularity> curl -L https://utexas.box.com/shared/static/hashvalue --output offline_data_gen/filtered-longhorn-v100.log
   ```


#### Training Data Generation (offline_data_gen)

   ```shell
   # Training data will be generated as batch_2800_7.pickle
   # The training data is from 2019-11-14T00:00:00 to 2021-02-27T00:00:00
   Singularity> unset PYTHONPATH
   Singularity> cd offline_data_gen
   Singularity> python3 offline_data_gen.py -parallel -num_samples 2802 -num_probe 7 -interval 4
   ```

#### Baseline generation (offline_data_gen)
   1. Change the init_time in offline_data_gent.py

   ```python
   # Replace Line 219 with the following init_time. 
   # The validation data is from 2021-03-01T00:00:00 to 2021-07-27T00:00:00
   simulator_init_time = datetime.strptime("2021-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
   ```

   2. Generating avg baseline and reactive baseline:
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
   Singularity> cd /dev/shm/Mirage/train_model/
   Singularity> mdkir data model

   # Copy the training data into "data" folder.
   Singularity> cp offline_data_gen/batch_2800_7.pickle data/

   # You may also download the training data from box. The data is valid till 05/14/2022
   Singularity> curl -L https://utexas.box.com/shared/static/2dnwd4v1uvp59spqza14x3xosoncvuki --output data/batch_2800_7.pickle

   # Train the model, the model will be model/model_2800_7.pt
   Singularity> python3 train.py
   ````

#### Online validation (online_validation)

   You can use the previously trained model or download the model from the provided link

1. Copy the log trace so that Ray can run in parallel
   ```shell
   Singularity> cd /dev/shm/Mirage/online_validation
   Singularity> cp ../offline_data_gen/filtered-longhorn-v100.log .
   ```

2. Use the trained model or download it from box
   ```shell
   # If using the trained model, you do not have to do anything
   # Otherwise, download the trained model using the following link
   curl -L https://utexas.box.com/shared/static/t3s3cdku866ly7hwyc0ysng4svqkjwpe --output /dev/shm/Mirage/train_model/model/model_2800_7.pt
   ```

3. Run
   ```shell
   # The output file will be reward.log under the same path
   Singularity> python3 top_validate.py -num_validate 888
   ```

4. You will see a reward.log that contains the validation rewards.
   

