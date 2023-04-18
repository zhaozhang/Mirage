import os
import pickle
import sys
import argparse
import numpy as np
import torch
import ray

from predictor import Predictor
from predictor import raw_features_to_np_ndarray

import random

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LINEAR_NEURAL_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,

    'n_hidden_units': [512, 256, 128, 64],
    'activation': torch.nn.LeakyReLU,
}

TRANSFORMER_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,

    'embed_size': 512,
    'encoder_nlayers': 2,
    'encoder_nheads': 2,
    'dim_feedforward': 128,
}

CONVOLUTION_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,
}

MODEL_NAME = 'transformer'
PARALLEL_PREPROCESSING = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-wd", "--work_dir", default="./")
    parser.add_argument("-n", "--file_name", required=True)
    parser.add_argument("-nd", "--data_name", required=True)
    parser.add_argument("-cpu_cores", default=None)
    parser.add_argument("-parallel", action="store_true", default=False)
    parser.add_argument("-num_experts", type=int, default=6)
    parser.add_argument("-mix_epoch", type=int, default=500)
    parser.add_argument("-expert_epoch", type=int, default=500)
    args = parser.parse_args()

    DATA_FILE_NAME = args.file_name
    PARALLEL_PREPROCESSING = args.parallel

    fname_data_raw = f'batch_{args.data_name}.pickle'
    fname_data_tensor = f'batch_{DATA_FILE_NAME}_cache.pickle'
    fname_model = f'model_{DATA_FILE_NAME}'

    if (not os.path.exists(os.path.join(args.work_dir, 'data', fname_data_tensor))):
        with open(os.path.join(args.work_dir, 'data', fname_data_raw), 'rb') as data:
            samples = pickle.load(data)
        raw_input = []
        raw_target = []
        samples = samples[:]
        for s in samples:
            if (s is not None):
                f, r = s
                raw_input.append(f)
                raw_target.append(r)

        if (PARALLEL_PREPROCESSING):
            try:
                if args.cpu_cores is not None:
                    ray.init(num_cpus=args.cpu_cores)
                else:
                    ray.init()
            except RuntimeError:
                print('Warning: Ray initialization failed.')
                sys.exit()
            np_ndarray_input = raw_features_to_np_ndarray(raw_input, parallel=True)

        else:
            np_ndarray_input = raw_features_to_np_ndarray(raw_input, parallel=False)

        np_ndarray_target = np.array(raw_target)
        tensor_input = torch.from_numpy(np_ndarray_input)
        tensor_target = torch.from_numpy(np_ndarray_target)
        tensor_target = tensor_target.reshape((tensor_target.shape[0], 1))
        tensor_target = tensor_target / 3600

        tensor_cache = (tensor_input, tensor_target)
        with open(os.path.join(args.work_dir, 'data', fname_data_tensor), 'wb') as output:
            pickle.dump(tensor_cache, output)
            print(f'{len(tensor_input)} tensorized samples are cached on local storage...')

    else:
        with open(os.path.join(args.work_dir, 'data', fname_data_tensor), 'rb') as data:
            tensor_input, tensor_target = pickle.load(data)
            print(f'{len(tensor_input)} tensorized samples are loaded on local cache...')

    data_size = tensor_input.shape[0]

    print("Training Start...", flush=True)
    EXPERTS_TRAIN_DATA_FRACTION = 6 / 10
    NUM_EXPERTS = args.num_experts
    TRAIN_DATA_FRACTION_PER_EXPERT = EXPERTS_TRAIN_DATA_FRACTION / NUM_EXPERTS

    EXPERT_TRAIN_SPLIT = 2 / 3
    EXPERT_TRAIN_BATCH_SIZE = 64
    EXPERT_VALIDATION_BATCH_SIZE = 1024
    EXPERT_TRAIN_EPOCHS = args.expert_epoch

    for iexpert in range(NUM_EXPERTS):

        if (os.path.exists(os.path.join(args.work_dir, '../..', fname_model + f'_expert_{iexpert}.pt'))):
            continue

        if (MODEL_NAME == 'linear'):
            predictor = Predictor(model_name='linear', hparams=LINEAR_NEURAL_REGRESSOR_CONFIG, optimizer_name='Adam',
                                  base_learning_rate=1e-4, device='gpu')
        elif (MODEL_NAME == 'convolution'):
            predictor = Predictor(model_name='convolution', hparams=CONVOLUTION_REGRESSOR_CONFIG, optimizer_name='Adam',
                                  base_learning_rate=1e-4, device='gpu')
        elif (MODEL_NAME == 'transformer'):
            predictor = Predictor(model_name='transformer', hparams=TRANSFORMER_REGRESSOR_CONFIG, optimizer_name='Adam',
                                  base_learning_rate=1e-4, device='gpu')
        else:
            predictor = None
            print(f'Error: Unsupported predictor type: {MODEL_NAME}!')
            sys.exit()

        expert_data_idx_start = int(data_size * iexpert * TRAIN_DATA_FRACTION_PER_EXPERT)
        expert_data_idx_end = int(data_size * (iexpert + 1) * TRAIN_DATA_FRACTION_PER_EXPERT)
        expert_tensor_input = tensor_input[expert_data_idx_start:expert_data_idx_end]
        expert_tensor_target = tensor_target[expert_data_idx_start:expert_data_idx_end]

        expert_data_size = expert_tensor_input.shape[0]
        assert (expert_data_size == expert_tensor_target.shape[0])
        rand_shuffle = torch.randperm(expert_data_size)
        expert_tensor_input = expert_tensor_input[rand_shuffle]
        expert_tensor_target = expert_tensor_target[rand_shuffle]

        expert_train_tensor_input = expert_tensor_input[:int(expert_data_size * EXPERT_TRAIN_SPLIT)]
        expert_train_tensor_target = expert_tensor_target[:int(expert_data_size * EXPERT_TRAIN_SPLIT)]
        expert_validation_tensor_input = expert_tensor_input[int(expert_data_size * EXPERT_TRAIN_SPLIT):]
        expert_validation_tensor_target = expert_tensor_target[int(expert_data_size * EXPERT_TRAIN_SPLIT):]

        iterator = predictor.batch_iterator(expert_train_tensor_input, expert_train_tensor_target,
                                            batch_size=EXPERT_TRAIN_BATCH_SIZE)
        for iepoch in range(EXPERT_TRAIN_EPOCHS):
            for iter, data in enumerate(iterator):
                data_input, data_target = data
                predictor.train_batch(data_input, data_target)

            validation_mean_l1loss = predictor.validate(expert_validation_tensor_input, expert_validation_tensor_target,
                                                        batch_size=EXPERT_VALIDATION_BATCH_SIZE)
            validation_mean_l1loss = round(validation_mean_l1loss, 2)
            print(f'Expert {iexpert} - MAE after Epoch {iepoch}: {validation_mean_l1loss} hours.', flush=True)

        model_checkpoint_path = os.path.join(args.work_dir, '../..', fname_model + f'_expert_{iexpert}.pt')
        predictor.save_checkpoint(model_checkpoint_path)
        del predictor

    '''
    print (torch.cuda.memory_allocated())
    sys.exit()
    '''

    MIXTURE_TRAIN_DATA_FRACTION = 3 / 10
    MIXTURE_TRAIN_DATA_SPLIT = 3 / 4
    mixture_data_start = int(data_size * EXPERTS_TRAIN_DATA_FRACTION)
    mixture_data_end = int(data_size * (EXPERTS_TRAIN_DATA_FRACTION + MIXTURE_TRAIN_DATA_FRACTION))
    mixture_tensor_input = tensor_input[mixture_data_start:mixture_data_end]
    mixture_tensor_target = tensor_target[mixture_data_start:mixture_data_end]
    mixture_data_size = len(mixture_tensor_input)
    rand_shuffle = torch.randperm(mixture_data_size)
    mixture_tensor_input = mixture_tensor_input[rand_shuffle]
    mixture_tensor_target = mixture_tensor_target[rand_shuffle]
    mixture_train_tensor_input = mixture_tensor_input[:int(mixture_data_size * MIXTURE_TRAIN_DATA_SPLIT)]
    mixture_train_tensor_target = mixture_tensor_target[:int(mixture_data_size * MIXTURE_TRAIN_DATA_SPLIT)]
    mixture_validation_tensor_input = mixture_tensor_input[int(mixture_data_size * MIXTURE_TRAIN_DATA_SPLIT):]
    mixture_validation_tensor_target = mixture_tensor_target[int(mixture_data_size * MIXTURE_TRAIN_DATA_SPLIT):]

    # TODO: Is cuda or cpu here?
    mixture_train_tensor_input = mixture_train_tensor_input.to('cpu')
    mixture_train_tensor_target = mixture_train_tensor_target.to('cpu')
    mixture_validation_tensor_input = mixture_validation_tensor_input.to('cpu')
    mixture_validation_tensor_target = mixture_validation_tensor_target.to('cpu')

    expert_list = []

    for iexpert in range(NUM_EXPERTS):
        checkpoint_path = os.path.join(args.work_dir, '../..', fname_model + f'_expert_{iexpert}.pt')
        print(f'Loading model: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model_name = checkpoint['model_name']
        model_hparams = checkpoint['model_hparams']
        model_state = checkpoint['model_state_dict']
        optimizer_name = checkpoint['optimizer_name']
        optimizer_state = checkpoint['optimizer_state_dict']
        expert_predictor = Predictor(model_name=model_name, hparams=model_hparams, optimizer_name=optimizer_name,
                                     model_state=model_state, optimizer_state=optimizer_state,
                                     base_learning_rate=1e-4, device='gpu')

        expert_list.append(expert_predictor.model)

    MOE_TRAIN_BATCH_SIZE = 32
    MOE_VALIDATION_BATCH_SIZE = 1024
    MOE_TRAIN_EPOCHS = args.mix_epoch
    nfeatures = mixture_train_tensor_input.shape[-1]
    seqlen = mixture_train_tensor_input.shape[-2]
    model_hparams = {'in_dim': nfeatures * seqlen, 'experts': expert_list}
    moe_predictor = Predictor(model_name='moe', hparams=model_hparams, optimizer_name='Adam', base_learning_rate=1e-3,
                              device='gpu')

    iterator = moe_predictor.batch_iterator(mixture_train_tensor_input, mixture_train_tensor_target,
                                            batch_size=MOE_TRAIN_BATCH_SIZE)
    for iepoch in range(MOE_TRAIN_EPOCHS):
        for iter, data in enumerate(iterator):
            data_input, data_target = data
            moe_predictor.train_batch(data_input, data_target)
            print(f'Mixture of Expert training... epoch {iepoch}:iter {iter}')

        validation_mean_l1loss = moe_predictor.validate(mixture_validation_tensor_input,
                                                        mixture_validation_tensor_target,
                                                        batch_size=MOE_VALIDATION_BATCH_SIZE)
        validation_mean_l1loss = round(validation_mean_l1loss, 2)
        print(f'Mixture of Expert - MAE after Epoch {iepoch}: {validation_mean_l1loss} hours.', flush=True)

    model_checkpoint_path = os.path.join(args.work_dir, '../..', f'moe_{DATA_FILE_NAME}.pt')
    moe_predictor.save_checkpoint(model_checkpoint_path)
