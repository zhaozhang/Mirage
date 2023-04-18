import os
import pickle
import sys

import numpy as np
import torch
import ray

from predictor import Predictor
from predictor import raw_features_to_np_ndarray

import random

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'gpu'

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
    'encoder_nlayers': 1,
    'encoder_nheads': 1,
    'dim_feedforward': 128,
}

CONVOLUTION_REGRESSOR_CONFIG = {
    'in_dim': 42,
    'seq_len': 144,
}

MODEL_NAME = 'transformer'
PARALLEL_PREPROCESSING = True
DATA_FILE_NAME = 'frontera_2682_7'

if __name__ == '__main__':

    fname_data_raw = f'batch_{DATA_FILE_NAME}.pickle'
    fname_data_tensor = f'batch_{DATA_FILE_NAME}_cache.pickle'
    fname_model = f'model_{DATA_FILE_NAME}'

    if (not os.path.exists(os.path.join('data', fname_data_tensor))):
        with open(os.path.join('data', fname_data_raw), 'rb') as data:
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
                ray.init(num_cpus=6)
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
        with open(os.path.join('data', fname_data_tensor), 'wb') as output:
            pickle.dump(tensor_cache, output)
            print(f'{len(tensor_input)} tensorized samples are cached on local storage...')

    else:
        with open(os.path.join('data', fname_data_tensor), 'rb') as data:
            tensor_input, tensor_target = pickle.load(data)
            print(f'{len(tensor_input)} tensorized samples are loaded on local cache...')

    data_size = tensor_input.shape[0]

    EXPERTS_TRAIN_DATA_FRACTION = 5 / 10
    NUM_EXPERTS = 10
    TRAIN_DATA_FRACTION_PER_EXPERT = EXPERTS_TRAIN_DATA_FRACTION / NUM_EXPERTS

    EXPERT_TRAIN_SPLIT = 2 / 3
    EXPERT_TRAIN_BATCH_SIZE = 64
    EXPERT_VALIDATION_BATCH_SIZE = 4096

    EXPERT_TRAIN_EPOCHS = [1000,
                           1500,
                           1500,
                           1500,
                           1500,
                           1500,
                           1500,
                           2000,
                           2000,
                           2000]
    assert (len(EXPERT_TRAIN_EPOCHS) == NUM_EXPERTS)

    EXPERT_TRAIN_LR = [1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       1e-5,
                       ]
    assert (len(EXPERT_TRAIN_LR) == NUM_EXPERTS)

    for iexpert in range(NUM_EXPERTS):

        if (os.path.exists(os.path.join('../..', fname_model + f'_expert_{iexpert}.pt'))):
            continue

        if (MODEL_NAME == 'linear'):
            predictor = Predictor(model_name='linear', hparams=LINEAR_NEURAL_REGRESSOR_CONFIG, optimizer_name='Adam',
                                  base_learning_rate=EXPERT_TRAIN_LR[iexpert], device=DEVICE)
        elif (MODEL_NAME == 'convolution'):
            predictor = Predictor(model_name='convolution', hparams=CONVOLUTION_REGRESSOR_CONFIG, optimizer_name='Adam',
                                  base_learning_rate=EXPERT_TRAIN_LR[iexpert], device=DEVICE)
        elif (MODEL_NAME == 'transformer'):
            predictor = Predictor(model_name='transformer', hparams=TRANSFORMER_REGRESSOR_CONFIG, optimizer_name='Adam',
                                  base_learning_rate=EXPERT_TRAIN_LR[iexpert], device=DEVICE)
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
        for iepoch in range(EXPERT_TRAIN_EPOCHS[iexpert]):
            for iter, data in enumerate(iterator):
                data_input, data_target = data
                predictor.train_batch(data_input, data_target)

            validation_l1loss = predictor.validate(expert_validation_tensor_input, expert_validation_tensor_target,
                                                   batch_size=EXPERT_VALIDATION_BATCH_SIZE)
            validation_mean_l1loss = round(np.mean(validation_l1loss), 2)
            validation_max_l1loss = round(np.max(validation_l1loss), 2)
            print(
                f'Mixture of Expert - Absoluate Error after Epoch {iepoch} - Max: {validation_max_l1loss} hours, Mean: {validation_mean_l1loss} hours')
            sys.stdout.flush()
            sys.stderr.flush()

        model_checkpoint_path = os.path.join('../..', fname_model + f'_expert_{iexpert}.pt')
        predictor.save_checkpoint(model_checkpoint_path)
        del predictor

    MIXTURE_TRAIN_DATA_FRACTION = 3 / 10
    mixture_train_start = 0
    mixture_train_end = int(data_size * (EXPERTS_TRAIN_DATA_FRACTION + MIXTURE_TRAIN_DATA_FRACTION))
    mixture_train_tensor_input = tensor_input[mixture_train_start:mixture_train_end]
    mixture_train_tensor_target = tensor_target[mixture_train_start:mixture_train_end]
    mixture_train_data_size = len(mixture_train_tensor_input)
    rand_shuffle = torch.randperm(mixture_train_data_size)
    mixture_train_tensor_input = mixture_train_tensor_input[rand_shuffle]
    mixture_train_tensor_target = mixture_train_tensor_target[rand_shuffle]
    mixture_train_tensor_input = mixture_train_tensor_input.to('cpu' if DEVICE == 'cpu' else 'cuda')
    mixture_train_tensor_target = mixture_train_tensor_target.to('cpu' if DEVICE == 'cpu' else 'cuda')

    moe_validation_start = int(data_size * (EXPERTS_TRAIN_DATA_FRACTION + MIXTURE_TRAIN_DATA_FRACTION))
    moe_validation_end = data_size
    moe_validation_tensor_input = tensor_input[moe_validation_start:moe_validation_end]
    moe_validation_tensor_target = tensor_target[moe_validation_start:moe_validation_end]
    moe_validation_data_size = len(moe_validation_tensor_input)
    rand_shuffle = torch.randperm(moe_validation_data_size)
    moe_validation_tensor_input = moe_validation_tensor_input[rand_shuffle]
    moe_validation_tensor_target = moe_validation_tensor_target[rand_shuffle]
    moe_validation_tensor_input = moe_validation_tensor_input.to('cpu' if DEVICE == 'cpu' else 'cuda')
    moe_validation_tensor_target = moe_validation_tensor_target.to('cpu' if DEVICE == 'cpu' else 'cuda')

    expert_list = []

    for iexpert in range(NUM_EXPERTS):
        checkpoint_path = os.path.join('../..', fname_model + f'_expert_{iexpert}.pt')
        print(f'Loading model: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model_name = checkpoint['model_name']
        model_hparams = checkpoint['model_hparams']
        model_state = checkpoint['model_state_dict']
        optimizer_name = checkpoint['optimizer_name']
        optimizer_state = checkpoint['optimizer_state_dict']
        expert_predictor = Predictor(model_name=model_name, hparams=model_hparams, optimizer_name=optimizer_name,
                                     model_state=model_state, optimizer_state=optimizer_state,
                                     base_learning_rate=1e-4, device=DEVICE)

        expert_list.append(expert_predictor.model)

    MOE_TRAIN_BATCH_SIZE = 128
    MOE_VALIDATION_BATCH_SIZE = 4096
    MOE_TRAIN_EPOCHS = 100
    MOE_TRAIN_MAX_ITERS = 1e10

    nfeatures = mixture_train_tensor_input.shape[-1]
    seqlen = mixture_train_tensor_input.shape[-2]
    model_hparams = {'in_dim': nfeatures * seqlen, 'experts': expert_list}
    moe_predictor = Predictor(model_name='moe', hparams=model_hparams, optimizer_name='Adam', base_learning_rate=1e-5,
                              device=DEVICE)

    iterator = moe_predictor.batch_iterator(mixture_train_tensor_input, mixture_train_tensor_target,
                                            batch_size=MOE_TRAIN_BATCH_SIZE)

    niters = 0

    for iepoch in range(MOE_TRAIN_EPOCHS):
        for iter, data in enumerate(iterator):
            data_input, data_target = data
            moe_predictor.train_batch(data_input, data_target)
            print(f'Mixture of Expert training... epoch {iepoch}:iter {iter}')
            niters += 1
            if (niters > MOE_TRAIN_MAX_ITERS):
                break

        validation_l1loss = moe_predictor.validate(moe_validation_tensor_input, moe_validation_tensor_target,
                                                   batch_size=MOE_VALIDATION_BATCH_SIZE)
        validation_mean_l1loss = round(np.mean(validation_l1loss), 2)
        validation_max_l1loss = round(np.max(validation_l1loss), 2)
        print(
            f'Mixture of Expert - Absoluate Error after Epoch {iepoch} - Max: {validation_max_l1loss} hours, Mean: {validation_mean_l1loss} hours')
        sys.stdout.flush()
        sys.stderr.flush()

        if (niters > MOE_TRAIN_MAX_ITERS):
            break

    model_checkpoint_path = os.path.join('../..', f'moe_{DATA_FILE_NAME}.pt')
    moe_predictor.save_checkpoint(model_checkpoint_path)
