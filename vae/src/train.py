# Train scanvi with custom parameters
import os
from typing import Any
import numpy as np
import scvi
from scvi.train import TrainRunner
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.model._utils import get_max_epochs_heuristic
import yaml
import scipy.sparse as sp
import anndata as ad
import logging

from ray import tune, train
import ray


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


class HPTuner:
    PERC_GPU: float = 1.0
    def __init__(
            self,
            adata: ad.AnnData,
            batch_label: str = 'dataset',
            cls_label: str = 'perturbation',
            perc_gpu_per_task: float = 1.0

    ):
        self.data_ref = ray.put(adata)
        self.batch_label: str = batch_label
        self.cls_label: str = cls_label
        self.PERC_GPU: float = perc_gpu_per_task

    def run_trial(self, params: dict) -> dict[str, Any]:
        metric = params.get('metric', 'validation_loss')
        model_params = params['model_params']
        data = ray.get(self.data_ref)
        data.X = sp.csr_matrix(data.X)
        logging.info(f'Setting up model with {params}')
        scvi.model.SCANVI.setup_anndata(data, batch_key=self.batch_label, labels_key=self.cls_label, unlabeled_category='unknown')
        model = scvi.model.SCANVI(data, **model_params)
        data_params = params['data_params']
        train_params = params['train_params']
        runner, data_splitter = prepare_scanvi(model, data_params, train_params)
        logging.info(f'Training model')
        runner()
        logging.info(f'Finished training model')
        metric_loss = runner.trainer.callback_metrics.get(metric)
        # train.report({metric: metric_loss})
        params['train_idx'] = data_splitter.train_idx
        params['val_idx'] = data_splitter.val_idx
        # logging.info(f'Saving model')
        # save_model(model, params, overwrite=True, save_anndata=False)
        return {metric: metric_loss}

    def run(self, search_space, 
            resources_per_trial: dict ={'cpu': 1, 'gpu': 1},
            num_samples: int = 12,
            metric: str = 'validation_loss',
            mode: str = 'min',
            output_dir: str = './'
        ):
        analysis = tune.run(
            self.run_trial,
            config=search_space,
            resources_per_trial=resources_per_trial,
            num_samples=num_samples,
            metric=metric,
            mode=mode,
            storage_path=output_dir,
            scheduler=tune.schedulers.ASHAScheduler(),
        )
        self.analysis = analysis
        return analysis
