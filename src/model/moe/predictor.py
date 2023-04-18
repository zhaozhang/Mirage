import numpy as np
import torch
import random
import sys
import torch.nn as nn
import sys
sys.path.append("../")
from moe.model import NeuralLinearRegressor, NerualConvRegressor, TransformerRegressor, MixtureOfExperts, \
    StochasticMixtureOfExperts, StochasticTransformer
import ray


class Predictor:
    def __init__(self,
                 model_name,
                 hparams,
                 optimizer_name,
                 base_learning_rate,
                 device,

                 model_state=None,
                 optimizer_state=None,

                 seed=0,
                 debug=True):

        self.set_seed(seed)

        self.model_name = model_name
        self.model_hparams = hparams

        if (self.model_name == 'linear'):
            self.model = NeuralLinearRegressor(
                in_dim=self.model_hparams['in_dim'] * self.model_hparams['seq_len'],
                n_hidden_units=self.model_hparams['n_hidden_units'],
                activation=self.model_hparams['activation']
            )
        elif (self.model_name == 'convolution'):
            self.model = NerualConvRegressor(in_dim=self.model_hparams['in_dim'],
                                             seq_len=self.model_hparams['seq_len'], )

        elif (self.model_name == 'transformer'):
            self.model = TransformerRegressor(in_size=self.model_hparams['in_dim'],
                                              seq_len=self.model_hparams['seq_len'],
                                              embed_size=self.model_hparams['embed_size'],
                                              encoder_nlayers=self.model_hparams['encoder_nlayers'],
                                              encoder_nheads=self.model_hparams['encoder_nheads'],
                                              dim_feedforward=self.model_hparams['dim_feedforward'])
        elif (self.model_name == "moe"):
            self.model = MixtureOfExperts(in_dim=self.model_hparams['in_dim'],
                                          expert_models=self.model_hparams['experts'])
        elif self.model_name == "moe_policy":
            self.model = StochasticMixtureOfExperts(in_dim=self.model_hparams['in_dim'],
                                                    expert_models=self.model_hparams['experts'])
        elif self.model_name == "transformer_policy":
            self.model = StochasticTransformer(in_dim=self.model_hparams['in_dim'],
                                               base_model=self.model_hparams['base'])
        else:
            print(f'Error: Model {self.model_name} is not supported!')
            sys.exit()

        if (model_state is not None):
            self.model.load_state_dict(model_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'gpu' else 'cpu')
        self.model.to(self.device)

        self.base_lr = base_learning_rate
        self.optimizer_name = optimizer_name
        if (self.optimizer_name == 'Adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)
        else:
            print(f'Error: Optimizer {optimizer_name} is not supported!')
            sys.exit()

        if (optimizer_state is not None):
            self.optimizer.load_state_dict(optimizer_state)

        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()

        if (debug):
            print(f'Model architecture: {self.model}')
            print(f'Model is placed on device: {next(self.model.parameters()).device}')

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)

    def train_batch(self, tensor_input, tensor_target):
        if (self.model_name == 'convolution'):
            tensor_input = torch.unsqueeze(tensor_input, 1)

        tensor_input = tensor_input.to(self.device)
        tensor_target = tensor_target.to(self.device)
        tensor_input, tensor_target = tensor_input.float(), tensor_target.float()
        self.optimizer.zero_grad()
        tensor_output = self.model(tensor_input)
        loss = self.mse_loss(tensor_output, tensor_target)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy().tolist()

    def inference(self, tensor_input):
        if (self.model_name == 'convolution'):
            tensor_input = torch.unsqueeze(tensor_input, 1)
        tensor_input = tensor_input.to(self.device)
        tensor_input = tensor_input.float()
        return self.model(tensor_input)

    def detach_tensor_to_numpy(self, t):
        return t.cpu().detach().numpy()

    def validate(self, tensor_input, tensor_target, batch_size):
        iterator = self.batch_iterator(tensor_input, tensor_target, batch_size)
        output_list = []
        target_list = []
        for iter, data in enumerate(iterator):
            data_input, data_target = data
            data_input = data_input.to(self.device).float()
            with torch.no_grad():
                output = self.model(data_input).cpu().detach()
                output_list.append(output)
                target = data_target.cpu().detach()
                target_list.append(target)
        validation_tensor_output = torch.concat(output_list, dim=0)
        validation_tensor_target = torch.concat(target_list, dim=0)
        l1_loss = self.l1_loss(validation_tensor_output, validation_tensor_target)
        return l1_loss.numpy().tolist()

    def batch_iterator(self, tensor_input, tensor_output, batch_size):
        class Samples(torch.utils.data.Dataset):
            def __init__(self, input, target):
                self.input = input
                self.target = target

            def __len__(self):
                return len(self.input)

            def __getitem__(self, idx):
                return self.input[idx], self.target[idx]

        iterator = torch.utils.data.DataLoader(Samples(tensor_input, tensor_output),
                                               batch_size=batch_size, shuffle=True, num_workers=0, )
        return iterator

    def save_checkpoint(self, checkpoint_path):
        try:
            torch.save({
                'model_name': self.model_name,
                'model_hparams': self.model_hparams,
                'model_state_dict': self.model.state_dict(),
                'optimizer_name': self.optimizer_name,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Model {self.model_name} is successfully checkpointed to {checkpoint_path}")
        except:
            print(f'Error: Failed checkpointing to path: {checkpoint_path}!')


@ray.remote
def raw_feature_to_np_ndarray_call_rpc(raw_sample):
    return raw_feature_to_np_ndarray_call(raw_sample)


def raw_feature_to_np_ndarray_call(raw_sample):
    n_steps = len(raw_sample)
    sample_features = []
    for i_step in range(n_steps):
        pds, rs, pred, succ = raw_sample[i_step]
        pred = list(pred)
        succ = list(succ)

        def get_percentiles(lst):
            return [np.percentile(lst, 0),
                    np.percentile(lst, 25),
                    np.percentile(lst, 50),
                    np.percentile(lst, 75),
                    np.percentile(lst, 100), ]

        npds = len(pds)
        if (npds > 0):
            pd_time_limits = [j[0] for j in pds]
            pd_nnodes = [j[1] for j in pds]
            pd_queue_t = [j[2] for j in pds]
            pd_runtime_t = [j[3] for j in pds]
        else:
            pd_time_limits = [0.0]
            pd_nnodes = [0.0]
            pd_queue_t = [0.0]
            pd_runtime_t = [0.0]

        nrs = len(rs)
        if (nrs > 0):
            r_time_limits = [j[0] for j in rs]
            r_nnodes = [j[1] for j in rs]
            r_queue_t = [j[2] for j in rs]
            r_runtime_t = [j[3] for j in rs]
        else:
            r_time_limits = [0.0]
            r_nnodes = [0.0]
            r_queue_t = [0.0]
            r_runtime_t = [0.0]

        step_features = []
        step_features += [npds]
        step_features += [sum(pd_nnodes)]
        step_features += get_percentiles(pd_time_limits)
        step_features += get_percentiles(pd_queue_t)
        step_features += get_percentiles(pd_runtime_t)

        step_features += [nrs]
        step_features += [sum(r_nnodes)]
        step_features += get_percentiles(r_time_limits)
        step_features += get_percentiles(r_queue_t)
        step_features += get_percentiles(r_runtime_t)

        step_features += pred
        step_features += succ

        sample_features.append(step_features)
    return sample_features


def raw_features_to_np_ndarray(raw_features, parallel=False):
    n_samples = len(raw_features)
    ndarray_features = []

    if (parallel):
        rpc_handles = []
        for i_sample in range(n_samples):
            print(f'Apply_async: formatting raw features of sample {i_sample}...')
            rpc_handles.append(raw_feature_to_np_ndarray_call_rpc.remote(raw_features[i_sample]))
        for i_sample, handle in enumerate(rpc_handles):
            ndarray_features.append(ray.get(handle))
            print(f'Raw features of sample {i_sample} is tensorized.')

    else:
        for i_sample in range(n_samples):
            # print(f'Foramatting raw features of sample {i_sample}...')
            ndarray_features.append(raw_feature_to_np_ndarray_call(raw_features[i_sample]))

    return np.array(ndarray_features)
