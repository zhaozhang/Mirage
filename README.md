## Mirage Source Code and User Manual

***

### Code Download

Download source code
```shell
git clone https://github.com/zhaozhang/Mirage.git
```

### Data Download

There are three types of data that we use in our project:

* job traces: derived from scheduler logs. The raw job trace is proprietary; please contact zzhang@tacc.utexas.edu to obtain a copy.
* please place the downloaded job trace files in src/workload.


### Manual

1. Create directory {MIRAGE_ROOT}/src/data and {MIRAGE_ROOT}/src/experiment
2. Copy {MIRAGE_ROOT}/src/model/moe to {MIRAGE_ROOT}/src/moe
3. Replace {MIRAGE_ROOT} with local mirage path
4. Execute following commands from offline data generation/baseline, model training, to model validation

Offline data generation (node 1)
```shell
cd {MIRAGE_ROOT}/src/top/
python3 offline_data_gen.py -parallel -num_samples 684 -num_probe 7 -interval 4 -od {MIRAGE_ROOT}/src/data/ls6/train_data_684 -workload filtered-ls6.log -start_time 2022-11-01T00:00:00 -warmup_len 2 -workload_len 5 -node 1 -baseline default
python3 {MIRAGE_ROOT}/script/pickle_merge.py -wd {MIRAGE_ROOT}/src/data/ls6/train_data_684 -out batch_ls6_684_7.pickle
```
Offline data generation (node 8)
```shell
cd {MIRAGE_ROOT}/src/top/
python3 offline_data_gen.py -parallel -num_samples 684 -num_probe 7 -interval 4 -od {MIRAGE_ROOT}/src/data/ls6/data_node_8_684 -workload filtered-ls6.log -start_time 2022-11-01T00:00:00 -warmup_len 2 -workload_len 5 -node 8 -baseline default
python3 {MIRAGE_ROOT}/script/pickle_merge.py -wd {MIRAGE_ROOT}/src/data/ls6/data_node_8_684 -out batch_ls6_684_7_node_8.pickle
```
Avg baseline (node 1)
```shell
cd {MIRAGE_ROOT}/src/top/
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_avg/baseline_avg -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 1 -baseline baseline_avg
python3 {MIRAGE_ROOT}/script/pickle_merge.py -wd {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_avg/baseline_avg -out baseline_avg_ls6_merge.pickle
```
Avg baseline (node 8)
```shell
cd {MIRAGE_ROOT}/src/top/
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_avg_node_8/baseline_avg -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 8 -baseline baseline_avg
python3 {MIRAGE_ROOT}/script/pickle_merge.py -wd {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_avg_node_8/baseline_avg -out baseline_avg_ls6_merge.pickle
```
Reactive baseline (node 1)
```shell
cd {MIRAGE_ROOT}/src/top/
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_reactive/baseline_reactive -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 1 -baseline baseline_reactive
python3 {MIRAGE_ROOT}/script/pickle_merge.py -wd {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_reactive/baseline_reactive -out baseline_reactive_ls6_merge.pickle
```
Reactive baseline (node 8)
```shell
cd {MIRAGE_ROOT}/src/top/
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_reactive_node_8/baseline_reactive -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 8 -baseline baseline_reactive
python3 {MIRAGE_ROOT}/script/pickle_merge.py -wd {MIRAGE_ROOT}/experiment/ls6_offline_gen_baseline_reactive_node_8/baseline_reactive -out baseline_reactive_ls6_merge.pickle
```
Train MoE
```shell
cd {MIRAGE_ROOT}/src/model/moe/
python3 train.py -wd ../../data/ls6/ -n moe_ls6 -parallel -nd ls6_684_7 -mix_epoch 300 -sample_window 144
```
Train xgboost
```shell
cd {MIRAGE_ROOT}/script/
python3 quantile_baseline.py -data /work/08377/dingqy/ls6/interrupt-free-provisioning/src/data/ls6/data/batch_moe_ls6_cache.pickle -out /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_xgboost/model/ls6_xgboost.pickle -config ./baseline_model_config.json -model RandomForest
```
Train transformer
```shell
cd {MIRAGE_ROOT}/src/model/init_version/
python3 train.py -wd ../../data/ls6/ -n transformer_ls6 -parallel -nd ls6_684_7 -epoch 300
```
Train MoE policy
```shell
cd {MIRAGE_ROOT}/src/model/policy-gradient-moe
python3 policy_gradient.py -train_cfg /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_moe_policy/train_cfg.json --use_cuda -config /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_moe_policy/sim.json -base_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_moe/model/ -base_prefix model_moe_ls6_expert -output_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_moe_policy/model/ -save 20 -num_experts 10
```
Train transformer policy
```shell
cd {MIRAGE_ROOT}/src/model/policy-gradient-transformer
python3 policy_gradient.py -train_cfg /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_transformer_policy/train_cfg.json --use_cuda -config /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_transformer_policy/sim.json -base_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_transformer/model/ -base_prefix model_transformer_ls6 -output_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_transformer_policy/model/ -save 60 
```
Train random forest
```shell
cd {MIRAGE_ROOT}/script/
python3 quantile_baseline.py -data /work/08377/dingqy/ls6/interrupt-free-provisioning/src/data/ls6/data/batch_moe_ls6_cache.pickle -out /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/ls6_train_random_forest/model/ls6_random_forest.pickle -config ./baseline_model_config.json -model RandomForest
```
Validate random forest
```shell
cd {MIRAGE_ROOT}/src/top
python3 online_validate.py -num_validate 156 -interval 4 -workload filtered-ls6.log -workload_len 5 -start_time 2023-03-01T00:00:00 -od {MIRAGE_ROOT}/experiment/ls6_validate_random_forest/result -m {MIRAGE_ROOT}/experiment/ls6_train_random_forest/model/ls6_random_forest.pickle -warmup_len 2 -sample_window 144 -parallel -mt baseline -node 1
```
Validate MoE
```shell
cd {MIRAGE_ROOT}/src/top
python3 online_validate.py -num_validate 156 -interval 4 -workload filtered-ls6.log -workload_len 5 -start_time 2023-03-01T00:00:00 -od {MIRAGE_ROOT}/experiment/ls6_validate_moe/result -m {MIRAGE_ROOT}/experiment/ls6_train_moe/model/moe_moe_ls6.pt -warmup_len 2 -sample_window 144 -parallel -mt moe -node 1
```
Validate MoE policy
```shell
cd {MIRAGE_ROOT}/src/top
python3 online_validate.py -num_validate 156 -interval 4 -workload filtered-ls6.log -workload_len 5 -start_time 2023-03-01T00:00:00 -od {MIRAGE_ROOT}/experiment/ls6_validate_moe_policy/result -m {MIRAGE_ROOT}/experiment/ls6_train_moe_policy/model/moe_policy_20.pt -warmup_len 2 -sample_window 144 -parallel -mt policy-gradient -node 1
```
Validate transformer
```shell
cd {MIRAGE_ROOT}/src/top
python3 online_validate.py -num_validate 156 -interval 4 -workload filtered-ls6.log -workload_len 5 -start_time 2023-03-01T00:00:00 -od {MIRAGE_ROOT}/experiment/ls6_validate_transformer/result -m {MIRAGE_ROOT}/experiment/ls6_train_transformer/model/model_transformer_ls6.pt -warmup_len 2 -sample_window 144 -parallel -mt moe -node 1
```
Validate transformer policy
```shell
cd {MIRAGE_ROOT}/src/top
python3 online_validate.py -num_validate 156 -interval 4 -workload filtered-ls6.log -workload_len 5 -start_time 2023-03-01T00:00:00 -od {MIRAGE_ROOT}/experiment/ls6_validate_transformer_policy/result -m {MIRAGE_ROOT}/experiment/ls6_train_transformer_policy/model/tx_policy_60.pt -warmup_len 2 -sample_window 144 -parallel -mt policy-gradient -node 1
```
Validate xgboost
```shell
cd {MIRAGE_ROOT}/src/top
python3 online_validate.py -num_validate 156 -interval 4 -workload filtered-ls6.log -workload_len 5 -start_time 2023-03-01T00:00:00 -od {MIRAGE_ROOT}/experiment/ls6_validate_xgboost/result -m {MIRAGE_ROOT}/experiment/ls6_train_xgboost/model/ls6_xgboost.pickle -warmup_len 2 -sample_window 144 -parallel -mt baseline -node 1
```
