## Pro2 Source Code and User Manual

***

### Code Download

Download source code
```shell
git clone https://github.com/zhaozhang/Pro_2.git
```

### Data Download

There are three types of data that we use in our project:

* job traces: derived from scheduler logs. The raw job trace is proprietary; please contact
  zzhang@tacc.utexas.edu if you want to get access.
* please place the downloaded job trace files in src/workload.


### Manual

Offline data generation (node 1)
```shell
python3 offline_data_gen.py -parallel -num_samples 684 -num_probe 7 -interval 4 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/data/ls6/data_node_8_684 -workload filtered-ls6.log -start_time 2022-11-01T00:00:00 -warmup_len 2 -workload_len 5 -node 1 -baseline default
python3 /work2/08377/dingqy/ls6/interrupt-free-provisioning/script/pickle_merge.py -wd /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/data/ls6/data_node_8_684 -out batch_ls6_684_7_node_8.pickle
```
Offline data generation (node 8)
```shell
python3 offline_data_gen.py -parallel -num_samples 684 -num_probe 7 -interval 4 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/data/ls6/data_node_8_684 -workload filtered-ls6.log -start_time 2022-11-01T00:00:00 -warmup_len 2 -workload_len 5 -node 8 -baseline default
python3 /work2/08377/dingqy/ls6/interrupt-free-provisioning/script/pickle_merge.py -wd /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/data/ls6/data_node_8_684 -out batch_ls6_684_7_node_8.pickle
```
Avg baseline (node 1)
```shell
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_avg/baseline_avg -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 1 -baseline baseline_avg
python3 /work2/08377/dingqy/ls6/interrupt-free-provisioning/script/pickle_merge.py -wd /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_avg/baseline_avg -out baseline_avg_ls6_merge.pickle
```
Avg baseline (node 8)
```shell
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_avg/baseline_avg -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 8 -baseline baseline_avg
python3 /work2/08377/dingqy/ls6/interrupt-free-provisioning/script/pickle_merge.py -wd /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_avg/baseline_avg -out baseline_avg_ls6_merge.pickle
```
Reactive baseline (node 1)
```shell
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_avg/baseline_avg -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 1 -baseline baseline_reactive
python3 /work2/08377/dingqy/ls6/interrupt-free-provisioning/script/pickle_merge.py -wd /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_reactive/baseline_reactive -out baseline_reactive_ls6_merge.pickle
```
Reactive baseline (node 8)
```shell
python3 offline_data_gen.py -parallel -num_samples 156 -num_probe 7 -interval 4 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_avg/baseline_avg -workload filtered-ls6.log -start_time 2023-03-01T00:00:00 -warmup_len 2 -workload_len 5 -node 8 -baseline baseline_reactive
python3 /work2/08377/dingqy/ls6/interrupt-free-provisioning/script/pickle_merge.py -wd /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_2/ls6_offline_gen_baseline_reactive/baseline_reactive -out baseline_reactive_ls6_merge.pickle
```
Train MoE
```shell
cd /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/model/moe/
python3 train.py -wd ../../data/ls6/ -n moe_ls6 -parallel -nd ls6_684_7 -mix_epoch 300 -sample_window 144
```
Train xgboost
```shell
python3 quantile_baseline.py -data /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/data/frontera/data/batch_frontera_2682_7_window_144_mix_300_cache.pickle -out /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_xgboost/model/frontera_xgboost.pickle -config ./baseline_model_config.json -model XGBRegression
```
Train transformer
```shell
cd /work2/08377/dingqy/ls6/interrupt-free-provisioning/src/model/init_version/
python3 train.py -wd ../../data/ls6/ -n transformer_ls6 -parallel -nd ls6_684_7 -epoch 300
```
Train MoE policy
```shell
python3 policy_gradient.py -train_cfg /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_moe_policy_node_8/train_cfg.json --use_cuda -config /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_moe_policy_node_8/sim.json -base_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_moe_node_8/model/ -base_prefix model_moe_frontera_node_8_expert -output_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_moe_policy_node_8/model/ -save 20 -num_experts 10
```
Train transformer policy
```shell
python3 policy_gradient.py -train_cfg /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_transformer_policy_node_8/train_cfg.json --use_cuda -config /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_transformer_policy_node_8/sim.json -base_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_transformer_node_8/model/ -base_prefix model_transformer_frontera_node_8 -output_dir /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_transformer_policy_node_8/model/ -save 60
```
Train random forest
```shell
python3 quantile_baseline.py -data /work/08377/dingqy/ls6/interrupt-free-provisioning/src/data/frontera/data/batch_frontera_2682_7_window_144_mix_300_cache.pickle -out /work/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_random_forest/model/frontera_random_forest.pickle -config ./baseline_model_config.json -model RandomForest
```
Validate random forest
```shell
python3 online_validate.py -num_validate 888 -interval 4 -workload filtered-frontera-rtx.log -workload_len 5 -start_time 2021-03-01T00:00:00 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_validate_random_forest/result -m /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_random_forest/model/frontera_random_forest.pickle -warmup_len 2 -sample_window 144 -parallel -mt baseline -node 1
```
Validate MoE
```shell
python3 online_validate.py -num_validate 888 -interval 4 -workload filtered-frontera-rtx.log -workload_len 5 -start_time 2021-03-01T00:00:00 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_validate_moe_node_8/result -m /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_moe_node_8/model/moe_moe_frontera_node_8.pt -warmup_len 2 -sample_window 144 -parallel -mt moe -node 8
```
Validate MoE policy
```shell
python3 online_validate.py -num_validate 888 -interval 4 -workload filtered-frontera-rtx.log -workload_len 5 -start_time 2021-03-01T00:00:00 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_validate_moe_policy_node_8/result -m /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_moe_policy_node_8/model/moe_policy_20.pt -warmup_len 2 -sample_window 144 -parallel -mt policy-gradient -node 8
```
Validate transformer
```shell
python3 online_validate.py -num_validate 888 -interval 4 -workload filtered-frontera-rtx.log -workload_len 5 -start_time 2021-03-01T00:00:00 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_validate_transformer_node_8/result -m /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_transformer_node_8/model/model_transformer_frontera_node_8.pt -warmup_len 2 -sample_window 144 -parallel -mt moe -node 8
```
Validate transformer policy
```shell
python3 online_validate.py -num_validate 888 -interval 4 -workload filtered-frontera-rtx.log -workload_len 5 -start_time 2021-03-01T00:00:00 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_validate_transformer_policy_node_8/result -m /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_transformer_policy_node_8/model/tx_policy_60.pt -warmup_len 2 -sample_window 144 -parallel -mt policy-gradient -node 8
```
Validate xgboost
```shell
python3 online_validate.py -num_validate 888 -interval 4 -workload filtered-frontera-rtx.log -workload_len 5 -start_time 2021-03-01T00:00:00 -od /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_validate_xgboost/result -m /work2/08377/dingqy/ls6/interrupt-free-provisioning/experiment/experiment_1/frontera_train_xgboost/model/frontera_xgboost.pickle -warmup_len 2 -sample_window 144 -parallel -mt baseline -node 8
```