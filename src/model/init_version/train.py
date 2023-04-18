import os
import pickle
import sys
import argparse
import numpy as np
import torch
import ray

from predictor import Predictor
from predictor import raw_features_to_np_ndarray

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

LSTM_CONDFIG = {
    'in_dim': 42,
    'hidden_size': 128,
    'num_layers': 1
}

MODEL_NAME = 'transformer'
PARALLEL_PREPROCESSING = True
TRAIN_FRACTION = 2 / 3
TRAIN_NEPOCHS = 200

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-wd", "--work_dir", default="./")
    parser.add_argument("-n", "--file_name", required=True)
    parser.add_argument("-nd", "--data_name", required=True)
    parser.add_argument("-cpu_cores", default=None)
    parser.add_argument("-parallel", action="store_true", default=False)
    parser.add_argument("-model", default="transformer")
    parser.add_argument("-epoch", type=int, default=500)
    args = parser.parse_args()

    TRAIN_NEPOCHS = args.epoch
    MODEL_NAME = args.model
    DATA_FILE_NAME = args.file_name
    PARALLEL_PREPROCESSING = args.parallel

    fname_data_raw = f'batch_{args.data_name}.pickle'
    fname_data_tensor = f'batch_{DATA_FILE_NAME}_cache.pickle'
    fname_model = f'model_{DATA_FILE_NAME}.pt'

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

    if (MODEL_NAME == 'linear'):
        predictor = Predictor(model_name='linear', hparams=LINEAR_NEURAL_REGRESSOR_CONFIG, optimizer_name='Adam',
                              base_learning_rate=1e-4, device='gpu')
    elif (MODEL_NAME == 'convolution'):
        predictor = Predictor(model_name='convolution', hparams=CONVOLUTION_REGRESSOR_CONFIG, optimizer_name='Adam',
                              base_learning_rate=1e-4, device='gpu')
    elif (MODEL_NAME == 'transformer'):
        predictor = Predictor(model_name='transformer', hparams=TRANSFORMER_REGRESSOR_CONFIG, optimizer_name='Adam',
                              base_learning_rate=1e-4, device='gpu')
    elif (MODEL_NAME == 'lstm'):
        predictor = Predictor(model_name='lstm', hparams=LSTM_CONDFIG, optimizer_name='Adam',
                              base_learning_rate=1e-4, device='gpu')
    else:
        predictor = None
        print(f'Error: Unsupported predictor type: {MODEL_NAME}!')
        sys.exit()

    data_size = tensor_input.shape[0]
    rand_shuffle = torch.randperm(data_size)
    tensor_input = tensor_input[rand_shuffle]
    tensor_target = tensor_target[rand_shuffle]

    train_size = int(data_size * TRAIN_FRACTION)
    test_size = int(data_size - train_size)
    assert (0 < train_size)
    assert (0 < test_size)
    train_tensor_input = tensor_input[:train_size]
    train_tensor_target = tensor_target[:train_size]
    test_tensor_input = tensor_input[train_size:]
    test_tensor_target = tensor_target[train_size:]

    iterator = predictor.batch_iterator(train_tensor_input, train_tensor_target, min_batch_size=128)
    for iepoch in range(TRAIN_NEPOCHS):
        for iter, data in enumerate(iterator):
            data_input, data_target = data
            predictor.train_batch(data_input, data_target)
            print(f'Complete training Iteration(Mini-batch)|Epoch: {iter}|{iepoch}')

        validation_mean_l1loss = predictor.validate(test_tensor_input, test_tensor_target, min_batch_size=128)
        validation_mean_l1loss = round(validation_mean_l1loss, 2)
        print(f'Mean l1 validation error after Epoch {iepoch}: {validation_mean_l1loss} hours.')

    model_checkpoint_path = os.path.join(args.work_dir, 'model', fname_model)
    predictor.save_checkpoint(model_checkpoint_path)
