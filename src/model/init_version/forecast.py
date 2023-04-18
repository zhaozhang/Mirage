import pickle
import numpy as np
import torch
import random
import sys

import torch.nn as nn

from model import TransformerRegressor

SEED = 100
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 16
NEPOCHS = 300

TRAIN_FRACTION = (0.1, 0.7)
VALIDATION_FRACTION = (0.7, 0.75)
LEARNING_RATE = 1e-4
EXPONENTIAL_LR_GAMMA=1.0-1e-3

SEQUENCE_LENGTH=6*24
WINDOW_SCALE=10
FUTURE_HORIZON=1
HORIZON_SCALE=10

CUDA_HOST_ALL_DATA=False
NEPOCHS_REPORT = 5



model = TransformerRegressor(in_size=1, seq_len=SEQUENCE_LENGTH,
                             embed_size=128, encoder_nlayers=2,
                             encoder_nheads=4, dim_feedforward=256)


with open('queue_state.pickle', 'rb') as data:
    queue_state_series = pickle.load(data)

with open('cluster_state.pickle', 'rb') as data:
    cluster_state_series = pickle.load(data)


queue_njobs_sequence=np.array([njobs for time, njobs in queue_state_series])
cluster_njobs_sequence=np.array([njobs for time, njobs in cluster_state_series])
assert(len(queue_njobs_sequence)==len(cluster_state_series))

def sliding_window_Xy(sequence,
                      window, window_scale,
                      horizon, horizon_scale):
    if(window<0):
        print('Error: Lengh of sliding window must be a positive integer.')
        return None
    seqlen = len(sequence)
    if(seqlen<=0):
        print('Warning: Null input sequence.')
        return sequence

    sequence_reshaped= np.array(sequence).reshape((seqlen, 1))
    window_start=0
    X=[]
    y=[]
    while(True):
        window_indice=[window_start + idx * window_scale for idx in range(window)]
        window_end=window_indice[-1]
        if(window_end>seqlen):
            break
        X.append(sequence_reshaped[window_indice])
        future=window_end+horizon*horizon_scale
        if(future>=seqlen):
            break
        y.append(sequence[future])
        window_start+=window_scale

    return np.array(X), np.array(y)

X,y=sliding_window_Xy(sequence=queue_njobs_sequence,
                      window=SEQUENCE_LENGTH, window_scale=WINDOW_SCALE,
                      horizon=FUTURE_HORIZON, horizon_scale=HORIZON_SCALE)




tensor_X = torch.from_numpy(X).float()
tensor_y = torch.from_numpy(y).float()

nsamples = tensor_X.shape[0]

training_samples_start = int(nsamples * TRAIN_FRACTION[0])
training_samples_end = int(nsamples * TRAIN_FRACTION[1])
validation_samples_start = int(nsamples * VALIDATION_FRACTION[0])
validation_samples_end = int(nsamples * VALIDATION_FRACTION[1])

training_X = tensor_X[training_samples_start:training_samples_end]
training_y = tensor_y[training_samples_start:training_samples_end]
validation_X = tensor_X[validation_samples_start:validation_samples_end]
validation_y = tensor_y[validation_samples_start:validation_samples_end]

ntraining_samples = len(training_X)
nvalidation_samples = len(validation_X)

print(f'INFO: Training set contains {ntraining_samples} samples.')
print(f'INFO: Validation set contains {nvalidation_samples} samples.')

if (ntraining_samples <= 0):
    print('Warning: Early termination, null training set.')
if (nvalidation_samples <= 0):
    print('Warning: Early termination, null validation set.')


if(CUDA_HOST_ALL_DATA):
    training_X = training_X.to(device)
    training_y = training_y.to(device)
    validation_X = validation_X.to(device)
    validation_y = validation_y.to(device)


mean_training = torch.mean(training_y).cpu().numpy()
var_training = torch.var(training_y).cpu().numpy()
std_training = torch.std(training_y).cpu().numpy()
print(f'Mean value for {len(training_y)}  training samples: {mean_training} jobs')
print(f'Std value for {len(training_y)}  training samples: {std_training} jobs')


mean_validation = torch.mean(validation_y).cpu().numpy()
var_validation = torch.var(validation_y).cpu().numpy()
std_validation = torch.std(validation_y).cpu().numpy()
print(f'Mean value for {len(validation_y)}  validation samples: {mean_validation} jobs')
print(f'Std value for {len(validation_y)}  validation samples: {std_validation} jobs')



class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


training_dataset = TorchDataset(X=training_X, y=training_y)
trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, )
validation_datatset = TorchDataset(X=validation_X, y=validation_y)
validation_loader = torch.utils.data.DataLoader(validation_datatset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, num_workers=0, )


model.to(device)
print(f'Model architecture: {model}')
print(f'Model is trained on device named: {next(model.parameters()).device}')

l1loss_mean = nn.L1Loss(reduction='mean')
l1loss_sum = nn.L1Loss(reduction='sum')
l2loss_mean = nn.MSELoss(reduction='mean')
l2loss_sum=nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=EXPONENTIAL_LR_GAMMA)

for epoch in range(1, NEPOCHS + 1):
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        inputs=inputs.to(device)
        targets=targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = l1loss_mean(outputs, targets)

        loss.backward()
        optimizer.step()

    lr_scheduler.step()

    print(f'Complete epoch {epoch}.')
    if (epoch % NEPOCHS_REPORT == 0):

        lr = optimizer.param_groups[0]["lr"]
        print(f'Learning rate after epoch {epoch}: {lr}')

        training_loss = 0.0
        squared_error = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = l1loss_sum(outputs, targets)
            training_loss += loss.item()

            squared_error += l2loss_sum(outputs, targets).item()

        print(
            f'MAE (Mean Absolute Error) on training set after epoch {epoch}: {round(training_loss / ntraining_samples, 3)} jobs')
        print(
            f'RMAE (Relative Mean Absolute Error) on training set after epoch {epoch}: {round(100 * (training_loss / ntraining_samples) / np.mean(y), 3)}%')

        mse= squared_error / ntraining_samples
        print(f'(MSE) Mean Squared Error on training set after epoch {epoch}: {mse}')
        rsquared=1.0-mse/var_training
        print(f'R-Squared on training set after epoch {epoch}:  {round(100*rsquared,2)}%')


        validation_loss = 0.0
        squared_error=0.0
        for i, data in enumerate(validation_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = l1loss_sum(outputs, targets)
            validation_loss += loss.item()

            squared_error+=l2loss_sum(outputs, targets).item()

        print(
            f'MAE (Mean Absolute Error) on validation set after epoch {epoch}: {round(validation_loss / nvalidation_samples, 3)} jobs')
        print(
            f'RMAE (Relative Mean Absolute Error) on validation set after epoch {epoch}: {round(100 * (validation_loss / nvalidation_samples) / mean_validation, 3)}%')

        mse = squared_error / nvalidation_samples
        print(f'(MSE) Mean Squared Error on validation set after epoch {epoch}: {mse}')
        rsquared = 1.0 - mse / var_validation
        print(f'R-Squared on validation set after epoch {epoch}: {round(100*rsquared,2)}%')

    sys.stdout.flush()
    sys.stderr.flush()

print('Training completes.')

