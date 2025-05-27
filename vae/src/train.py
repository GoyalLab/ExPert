# Train scanvi with custom parameters
import os
from typing import Any
import numpy as np
import scvi
from scvi.train import TrainRunner
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.model._utils import get_max_epochs_heuristic
import yaml


def prepare_scanvi(model: scvi.model.SCANVI, data_params: dict[str, Any]={}, scanvi_params: dict[str, Any]={}, train_params: dict[str, Any]={}):
    epochs = train_params.get('max_epochs')
    # determine number of epochs needed for complete training
    max_epochs = get_max_epochs_heuristic(model.adata.n_obs)
    if epochs is None:
        if model.was_pretrained:
            max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))
        epochs = max_epochs
    print(f'Max epochs suggested: {max_epochs}')

    train_size: int = data_params.pop('train_size', 0.9)
    if not train_size < 1.0 and train_size > 0:
        raise ValueError(f'Parameter train_size should be between 0 and 1, got {train_size}')
    
    #data_params['validation_size'] = (100 - int(train_size * 100)) / 100

    # create data splitter
    data_splitter = SemiSupervisedDataSplitter(
        adata_manager=model.adata_manager,
        **data_params,
    )
    plan_kwargs = train_params.pop('plan_kwargs', {})
    # create training plan
    training_plan = model._training_plan_cls(model.module, model.n_labels, **plan_kwargs)
    check_val_every_n_epoch = train_params.pop('check_val_every_n_epoch', 10)
    # create training runner
    runner = TrainRunner(
        model,
        training_plan=training_plan,
        data_splitter=data_splitter,
        accelerator='auto',
        devices='auto',
        check_val_every_n_epoch=check_val_every_n_epoch,
        **train_params
    )
    if 'logger' in train_params.keys():
        # save hyper-parameters to lightning logs
        hparams = {
            'data_params': data_params,
            'scanvi_params': scanvi_params,
            'plan_params': plan_kwargs,
            'train_params': train_params
        }
        runner.trainer.logger.log_hyperparams(hparams)
    return runner, data_splitter


def get_lightning_log_dir(logger):
    return os.path.join(logger._root_dir, logger._name, f'version_{logger._version}')


def get_model_output(model_dir):
    run_idc = []
    for el in os.listdir(model_dir):
        e = os.path.join(model_dir, el)
        if os.path.isdir(e) and el.startswith('run_'):
            run_idc.append(int(el.split('_')[1]))
    if len(run_idc) == 0:
        idx = 0
    else:
        idx = np.max(run_idc)+1
    f = os.path.join(model_dir, f'run_{idx}')
    return f


def save_params(params, model_dir):
    params_path = os.path.join(model_dir, 'hparams.yaml')
    with open(params_path, 'w') as file:
        yaml.dump(params, file)
    

def save_model(model, params={}, **save_kwargs):
    # get lightning log directory and save additional data to it
    m = get_lightning_log_dir(params['train_params'].get('logger'))
    # save all associated parameters to file
    save_params(params, m)
    # save model
    model.save(m, **save_kwargs)  
