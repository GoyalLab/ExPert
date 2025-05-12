
from src.modules._genevae import GENEVAE
from src.utils.constants import REGISTRY_KEYS
from src._train.plan import SemiSupervisedTrainingPlan

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from sklearn.utils.class_weight import compute_class_weight


from collections.abc import Sequence
from typing import Literal, Any, Iterable

from anndata import AnnData

# scvi imports
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.data import AnnDataManager
from scvi.model._utils import _init_library_size, get_max_epochs_heuristic
from scvi.train import TrainRunner
from scvi.data._utils import get_anndata_attribute
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
from scvi.train._callbacks import SubSampleLabels
from scvi.model.base import (
    BaseModelClass,
    ArchesMixin,
    VAEMixin,
    RNASeqMixin
)
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    ObsmField,
    NumericalJointObsField,
    NumericalObsField,
)

logger = logging.getLogger(__name__)


class GENEVI(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    _module_cls = GENEVAE
    _training_plan_cls = SemiSupervisedTrainingPlan
    _LATENT_QZM_KEY = "genevi_latent_qzm"
    _LATENT_QZV_KEY = "genevi_latent_qzv"


    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        dropout_rate_decoder: float = 0.2,
        dropout_rate_encoder: float = 0.2,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        empirical_gene_background_prior: bool | None = None,
        linear_classifier: bool = False,
        use_class_weights: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        genevae_model_kwargs = dict(model_kwargs)
        # Initialize indices and labels for this VAE
        self._set_indices_and_labels()

        # Ignores unlabeled catgegory
        n_labels = self.summary_stats.n_labels - 1
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        # Add classification weights
        class_weights = None
        if use_class_weights:
            labels_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
            labels = self.adata.obs[labels_key].values
            class_weights = compute_class_weight('balanced', classes=self._label_mapping[:-1], y=labels)
        # Determine number of batches/datasets and fix library method
        n_batch = self.summary_stats.n_batch
        # We always use observed library sizes, so no need for that stuff here
        # library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        emp_prior = (
            empirical_gene_background_prior
            if empirical_gene_background_prior is not None
            else (self.summary_stats.n_gene_embedding > 10)
        )
        if emp_prior:
            # TODO: add prior calculation here and understand if its needed in out case (might not)
            prior_mean, prior_scale = None, None
        else:
            prior_mean, prior_scale = None, None

        # Initialize genevae
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_input_emb=self.summary_stats.n_gene_embedding,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            dropout_rate_encoder=dropout_rate_encoder,
            dropout_rate_decoder=dropout_rate_decoder,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            linear_classifier=linear_classifier,
            emb_background_prior_mean=prior_mean,
            emb_background_prior_scale=prior_scale,
            **genevae_model_kwargs,
        )

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.n_labels = n_labels
        # TODO: give more detailed summary
        self._model_summary_string = (
            f"GeneVI Model with the following params: \n"
            f"unlabeled_category: {self.unlabeled_category_}, n_classes: {self.n_labels}, "
            f"dispersion: {dispersion}, gene_likelihood: {gene_likelihood}"
        )

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells. Same as used in scanVI.x"""
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        # set unlabeled and labeled indices
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        soft: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        indices
            Return probabilities for each class label.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        use_posterior_mean
            If ``True``, uses the mean of the posterior distribution to predict celltype
            labels. Otherwise, uses a sample from the posterior distribution - this
            means that the predictions will be stochastic.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )
        y_pred = []
        for _, tensors in enumerate(scdl):
            x = tensors[REGISTRY_KEYS.X_KEY]
            g = tensors[REGISTRY_KEYS.GENE_EMB_KEY]
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            pred = self.module.classify(
                x,
                g,
                batch_index=batch,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
            )
            if self.module.classifier.logits:
                pred = torch.nn.functional.softmax(pred, dim=-1)
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred).numpy()
        if not soft:
            predictions = []
            for p in y_pred:
                predictions.append(self._code_to_label[p])

            return np.array(predictions)
        else:
            n_labels = len(pred[0])
            pred = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
            return pred
        
    @devices_dsp.dedent
    def train(
        self,
        data_params: dict[str, Any]={}, 
        model_params: dict[str, Any]={}, 
        train_params: dict[str, Any]={},
        return_runner: bool = True
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset for semisupervised training.
        n_samples_per_label
            Number of subsamples for each label class to sample per epoch. By default, there
            is no label subsampling.
        check_val_every_n_epoch
            Frequency with which metrics are computed on the data for validation set for both
            the unsupervised and semisupervised trainers. If you'd like a different frequency for
            the semisupervised trainer, set check_val_every_n_epoch in semisupervised_train_kwargs.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        %(param_accelerator)s
        %(param_devices)s
        datasplitter_kwargs
            Additional keyword arguments passed into
            :class:`~scvi.dataloaders.SemiSupervisedDataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.SemiSupervisedTrainingPlan`. Keyword arguments
            passed to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        epochs = train_params.get('max_epochs')
        # determine number of epochs needed for complete training
        max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
        if epochs is None:
            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))
            epochs = max_epochs
        logging.info(f'Epochs suggested: {max_epochs}, training for {epochs} epochs.')

        train_size: int = data_params.pop('train_size', 0.9)
        if not train_size < 1.0 and train_size > 0:
            raise ValueError(f'Parameter train_size should be between 0 and 1, got {train_size}')

        # Create data splitter
        data_splitter = SemiSupervisedDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            **data_params,
        )

        plan_kwargs = train_params.pop('plan_kwargs', {})
        # create training plan
        training_plan = self._training_plan_cls(self.module, self.n_labels, **plan_kwargs)
        check_val_every_n_epoch = train_params.pop('check_val_every_n_epoch', 10)
        # if we have labeled cells, we want to subsample labels each epoch
        sampler_callback = [SubSampleLabels()] if len(self._labeled_indices) != 0 else []
        if "callbacks" in train_params.keys():
            train_params["callbacks"] += [sampler_callback]
        else:
            train_params["callbacks"] = sampler_callback
        # create training runner
        runner = TrainRunner(
            self,
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
                'scanvi_params': model_params,
                'plan_params': plan_kwargs,
                'train_params': train_params
            }
            runner.trainer.logger.log_hyperparams(hparams)
        # Train model
        if return_runner:
            return runner
        else:
            return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        unlabeled_category: str,
        layer: str | None = None,
        gene_emb_obsm_key: str | None = 'gene_embedding',
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        cast_to_csr: bool = True,
        raw_counts: bool = False,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Setup AnnData object for training a GENEVI model.
        """
        if not isinstance(adata.X, sp.csr_matrix) and cast_to_csr:
            logging.info('Casting adata.X to csr matrix to boost training efficiency.')
            adata.X = sp.csr_matrix(adata.X)
        setup_method_args = cls._get_setup_method_args(**locals())
        
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=raw_counts),
            ObsmField(REGISTRY_KEYS.GENE_EMB_KEY, gene_emb_obsm_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # Create AnnData manager
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
