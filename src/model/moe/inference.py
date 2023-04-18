import os
import torch

from train import Predictor


if(__name__=='__main__'):
    checkpoint_path=os.path.join('model', 'model_2000_7.pt')
    checkpoint=torch.load(checkpoint_path)

    model_name=checkpoint['model_name']
    model_hparams=checkpoint['model_hparams']
    model_state=checkpoint['model_state_dict']
    optimizer_name=checkpoint['optimizer_name']
    optimizer_state=checkpoint['optimizer_state_dict']

    predictor = Predictor(model_name=model_name, hparams=model_hparams, optimizer_name=optimizer_name,
                          model_state=model_state, optimizer_state=optimizer_state,
                          base_learning_rate=1e-4, device='gpu')



    toy_sample=torch.randn(10,144,42)
    pred_tensor=predictor.inference(toy_sample).squeeze(1)
    pred_numpy=predictor.detach_tensor_to_numpy(pred_tensor)

    print(pred_numpy)