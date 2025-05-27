import scipy.sparse as sp
import anndata as ad
import logging
import scvi
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from typing import Any

from ray import tune, train
import ray

from ._statics import HP_KEYS


class HPTuner:
    PERC_GPU: float = 1.0
    def __init__(
            self,
            adata: ad.AnnData,
            model_class: BaseModelClass,
            batch_label: str = 'dataset',
            cls_label: str = 'perturbation',
            perc_gpu_per_task: float = 1.0,

    ):
        self.data_ref = ray.put(adata)
        self.model_class = model_class
        self.batch_label: str = batch_label
        self.cls_label: str = cls_label
        self.PERC_GPU: float = perc_gpu_per_task

    def run_trial(self, params: dict) -> dict[str, Any]:
        metric = params.get('metric', 'validation_classification_loss')
        # Get individual params
        model_params = params[HP_KEYS.MODEL_PARAMS_KEY]
        data_params = params[HP_KEYS.DATA_PARAMS_KEY]
        train_params = params[HP_KEYS.TRAIN_PARAMS_KEY]

        data = ray.get(self.data_ref)
        logging.info(f'Setting up model with {params}')
        self.model_class.setup_anndata(data, labels_key=self.cls_label, batch_key=self.batch_label, unlabeled_category=HP_KEYS.UNKNOWN_CAT_KEY)
        # Initialize model
        model = self.model_class(data, **model_params)
        # Train model
        runner = model.train(
            data_params=data_params.copy(), 
            model_params=model_params.copy(), 
            train_params=train_params.copy(), 
            return_runner=True
        )
        logging.info(f'Training model')
        runner()
        logging.info(f'Finished training model')
        metric_loss = runner.trainer.callback_metrics.get(metric)
        # train.report({metric: metric_loss})
        # logging.info(f'Saving model')
        # save_model(model, params, overwrite=True, save_anndata=False)
        return {metric: metric_loss}

    def run(self, search_space, 
            resources_per_trial: dict ={'cpu': 1, 'gpu': 1},
            num_samples: int = 12,
            metric: str = 'elbo_validation',
            mode: str = 'min',
            output_dir: str = './tune'
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