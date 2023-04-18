import pickle
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as torch_functional
import random
import sys

SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
random.seed(SEED)
np.random.seed(SEED)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_BATCH_SIZE=64
VALIDATION_BATCH_SIZE=128
NEPOCHS=1000



with open('samples.pickle', 'rb') as data:
    samples= pickle.load(data)

X=np.array([s[0] for s in samples])
y=np.array([s[1] for s in samples])

tensor_X=torch.from_numpy(X)
tensor_X=torch.unsqueeze(tensor_X, 1)
tensor_y=torch.from_numpy(y)

nsamples=tensor_X.shape[0]
training_fraction=0.6
validation_fraction=0.1
ntraining_samples=int(nsamples*training_fraction)
nvalidation_samples=int(nsamples*validation_fraction)


training_X=tensor_X[:ntraining_samples]
training_y=tensor_y[:ntraining_samples]
validation_X=tensor_X[ntraining_samples:ntraining_samples+nvalidation_samples]
validation_y=tensor_y[ntraining_samples:ntraining_samples+nvalidation_samples]

training_X=training_X.to(device)
training_y=training_y.to(device)
validation_X=validation_X.to(device)
validation_y=validation_y.to(device)

mean_validation=torch.mean(validation_y).cpu().numpy()
print(f'Mean queueing delay for {len(validation_y)}  validation jobs in data set: {mean_validation} Hour')

class TorchDataset(torch.utils.data.Dataset):
  def __init__(self, X, y):
      self.X = X
      self.y = y
  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

training_dataset=TorchDataset(X=training_X, y=training_y)
trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0,)
validation_datatset=TorchDataset(X=validation_X, y=validation_y)
validation_loader = torch.utils.data.DataLoader(validation_datatset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True, num_workers=0,)





model=NNPredictor()
model.to(device)
print(f'Model architecture: {model}')
print(f'Model is trained on device named: {next(model.parameters()).device}')

l1loss_mean=nn.L1Loss(reduction='mean')
l1loss_sum=nn.L1Loss(reduction='sum')
optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)
l2loss=nn.MSELoss()


for epoch in range(1, NEPOCHS+1):
    print('-'*80)
    print(f'Epoch {epoch}')
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = l1loss_mean(outputs, targets)

        loss.backward()
        optimizer.step()

    if(epoch%10==0):
        training_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = model(inputs)
            loss = l1loss_sum(outputs, targets)
            training_loss+=loss.item()
        print(
            f'MAE (Mean Absolute Error) on training set after epoch {epoch}: {round(3600 * training_loss / ntraining_samples, 3)} seconds')
        print(
            f'RMAE (Relative Mean Absolute Error) on training set after epoch {epoch}: {round(100 * (training_loss / ntraining_samples) / np.mean(y), 3)}%')

        validation_loss=0.0
        for i, data in enumerate(validation_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = model(inputs)
            loss = l1loss_sum(outputs, targets)
            validation_loss+=loss.item()
        print(f'MAE (Mean Absolute Error) on validation set after epoch {epoch}: {round(3600 * validation_loss / nvalidation_samples, 3)} seconds')
        print(f'RMAE (Relative Mean Absolute Error) on validation set after epoch {epoch}: {round(100 * (validation_loss / nvalidation_samples) / mean_validation, 3)}%')


print('Training completes.')

