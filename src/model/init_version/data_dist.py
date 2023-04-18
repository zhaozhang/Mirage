import pickle
import numpy as np
import pandas as pd
import torch
import random
import sys

from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





print('Analyzing qtpred.pickle.....')
with open('qtpred.pickle', 'rb') as data:
    samples = pickle.load(data)

X = []
y = []
for i, s in enumerate(samples):
    if (s is not None):
        X.append(s[0])
        y.append(s[1])
    else:
        print('Warning: Check integrity...skip null record')
X = np.array(X)
y = np.array(y)
tensor_X = X
tensor_y = y

TRAIN_FRACTION = (0.1, 0.4)
VALIDATION_FRACTION = (0.4, 0.7)
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

print(f'Two-sample Kolmogorov-Smirnov test for Training-Validation y: {ks_2samp(training_y, validation_y)}')

df_training_y=pd.DataFrame({'training_y （queueing time)':pd.Series(training_y)})
print(df_training_y.describe(include='all'))

df_validation_y=pd.DataFrame({'validation_y (queueing time)':pd.Series(validation_y)})
print(df_validation_y.describe(include='all'))