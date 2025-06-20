
from src.modules._jedvae import JEDVAE
from src.modules._splitter import ContrastiveDataSplitter, SemiSupervisedDataSplitter
from src.utils.constants import REGISTRY_KEYS
from src._train.plan import SemiSupervisedTrainingPlan, ContrastiveSupervisedTrainingPlan
from src.utils.preprocess import _prep_adata
from src.data._manager import EmbAnnDataManager

import torch
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from sklearn.utils.class_weight import compute_class_weight


from collections.abc import Sequence
from typing import Literal, Any

from anndata import AnnData

# scvi imports
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner
from scvi.data._manager import AnnDataManager
from scvi.data._utils import get_anndata_attribute, _check_if_view
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
from scvi.train._callbacks import SubSampleLabels
from scvi.model.base import (
    BaseModelClass,
    ArchesMixin,
    VAEMixin,
    RNASeqMixin,
    UnsupervisedTrainingMixin
)
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    StringUnsField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi._types import AnnOrMuData
from scvi.model._scvi import SCVI

logger = logging.getLogger(__name__)


class JEDVI(
        RNASeqMixin, 
        VAEMixin,
        ArchesMixin, 
        BaseModelClass,
        UnsupervisedTrainingMixin,
    ):
    _module_cls = JEDVAE
    _training_plan_cls = ContrastiveSupervisedTrainingPlan
    _LATENT_QZM_KEY = "gpvi_latent_qzm"
    _LATENT_QZV_KEY = "gpvi_latent_qzv"


    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        dropout_rate_decoder: float | None = None,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        linear_classifier: bool = False,
        cls_weight_method: str | None = "balanced",
        **model_kwargs,
    ):
        super().__init__(adata)
        _model_kwargs = dict(model_kwargs)
        self._model_kwargs = _model_kwargs
        # Initialize indices and labels for this VAE
        self._set_indices_and_labels()

        # Ignores unlabeled catgegory
        n_labels = self.summary_stats.n_labels - 1
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        # Add classification weights based on support in data set
        cls_weights = self._get_cls_weights(cls_weight_method)
        # Determine number of batches/datasets and fix library method
        n_batch = self.summary_stats.n_batch

        # Initialize genevae
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers_encoder=n_layers,
            n_layers_decoder=n_layers,
            dropout_rate_encoder=dropout_rate,
            dropout_rate_decoder=dropout_rate_decoder,
            class_embed_dim=self.n_dims_emb,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            linear_classifier=linear_classifier,
            cls_weights=cls_weights,
            **_model_kwargs,
        )

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.n_labels = n_labels
        # TODO: give more detailed summary
        self._model_summary_string = (
            f"{self.__class__} Model with the following params: \n"
            f"n_classes: {self.n_labels}, "
        )

    def _get_cls_weights(
        self,
        method: str | None = 'balanced'
    ) -> torch.Tensor | None:
        if method is None:
            return None
        else:
            labels_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
            labels = self.adata.obs[labels_key].values
            class_weights = compute_class_weight(method, classes=self._label_mapping[:-1], y=labels)
            return torch.tensor(class_weights, dtype=torch.float32)

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells. Prepare class embedding"""
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
        # order class embedding according to label mapping and transform to matrix
        if REGISTRY_KEYS.CLS_EMB_KEY in self.adata.uns:
            if 'CLS_EMB_INIT' not in self.adata.uns:
                cls_emb: pd.DataFrame = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY]
                if not isinstance(cls_emb, pd.DataFrame):
                    logging.warning(f'Class embedding has to be a dataframe with labels as index, got {cls_emb.__class__}. Falling back to internal embedding.')
                else:
                    # Order external embedding to match label mapping and convert to csr matrix
                    _label_series = pd.Series(self._code_to_label.values())[:-1]
                    _label_overlap = _label_series.isin(self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY].index)
                    _shared_labels = _label_series[_label_overlap].values
                    n_missing = _label_overlap.shape[0] - _label_overlap.sum()
                    if n_missing > 0:
                        raise ValueError(f'Found {n_missing} missing labels in cls embedding: {_label_series[~_label_overlap]}')
                    self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY] = sp.csr_matrix(self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY].loc[_shared_labels].values)
                    self.adata.uns['CLS_EMB_INIT'] = {'n_labels': _shared_labels.shape[0]}
            else:
                logging.info(f'Class embedding has already been initialized with {self.__class__} for this adata.')
            self.n_dims_emb = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY].shape[1]
        else:
            logging.info(f'No class embedding found in adata, falling back to internal embeddings with dimension 128. You can change this by specifying `n_dims_emb`.')
            self.n_dims_emb = self._model_kwargs.get('n_dims_emb', 128)

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **model_kwargs,
    ):
        """Initialize scanVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~GEDVI.setup_anndata`.
        model_kwargs
            kwargs for gedvi model
        """
        from copy import deepcopy
        from scvi import settings
        from scvi.data._constants import (
            _SETUP_ARGS_KEY,
            ADATA_MINIFY_TYPE,
        )
        from scvi.data._utils import _is_minified

        scvi_model._check_if_trained(message="Passed in scvi model hasn't been trained yet.")

        model_kwargs = dict(model_kwargs)
        init_params = scvi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in model_kwargs.keys():
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"SCVI model with value {v}.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                del model_kwargs[k]

        if scvi_model.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise ValueError(
                f"We cannot use the given scVI model to initialize {cls.__name__} because it has "
                "minified adata. Keep counts when minifying model using "
                "minified_data_type='latent_posterior_parameters_with_counts'."
            )

        if adata is None:
            adata = scvi_model.adata
        else:
            if _is_minified(adata):
                raise ValueError(f"Please provide a non-minified `adata` to initialize {cls.__name__}.")
            # validate new anndata against old model
            scvi_model._validate_anndata(adata)

        scvi_setup_args = deepcopy(scvi_model.adata_manager.registry[_SETUP_ARGS_KEY])
        scvi_labels_key = scvi_setup_args["labels_key"]
        if labels_key is None and scvi_labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the scVI model was initialized without one."
            )
        if scvi_labels_key is None:
            scvi_setup_args.update({"labels_key": labels_key})
        # Setup adata
        cls.setup_anndata(
            adata,
            unlabeled_category=unlabeled_category,
            **scvi_setup_args,
        )
        gedvi_model = cls(adata, **non_kwargs, **kwargs, **model_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        gedvi_model.module.load_state_dict(scvi_state_dict, strict=False)
        gedvi_model.was_pretrained = True

        return gedvi_model
    
    @classmethod
    def from_base_model(
        cls,
        scvi_model,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **model_kwargs,
    ):
        """Initialize scanVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~GEDVI.setup_anndata`.
        model_kwargs
            kwargs for gedvi model
        """
        from copy import deepcopy
        from scvi import settings
        from scvi.data._constants import (
            _SETUP_ARGS_KEY,
        )
        from scvi.data._utils import _is_minified

        scvi_model._check_if_trained(message="Passed in scvi model hasn't been trained yet.")

        model_kwargs = dict(model_kwargs)
        init_params = scvi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in model_kwargs.keys():
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"SCVI model with value {v}.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                del model_kwargs[k]

        if adata is None:
            adata = scvi_model.adata
        else:
            if _is_minified(adata):
                raise ValueError(f"Please provide a non-minified `adata` to initialize {cls.__name__}.")
            # validate new anndata against old model
            scvi_model._validate_anndata(adata)

        scvi_setup_args = deepcopy(scvi_model.adata_manager.registry[_SETUP_ARGS_KEY])
        scvi_labels_key = scvi_setup_args["labels_key"]
        if labels_key is None and scvi_labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the scVI model was initialized without one."
            )
        if scvi_labels_key is None:
            scvi_setup_args.update({"labels_key": labels_key})
        cls.setup_anndata(
            adata,
            **scvi_setup_args,
        )
        model = cls(adata, **non_kwargs, **kwargs, **model_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        model.module.load_state_dict(scvi_state_dict, strict=False)
        model.was_pretrained = True

        return model

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
            AnnData object that has been registered via :meth:`~GEDVI.setup_anndata`.
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
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]
            ext_cls_emb = tensors.get(REGISTRY_KEYS.CLS_EMB_KEY)

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            pred, _ = self.module.classify(
                x,
                batch_index=batch,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
                class_embeds=ext_cls_emb
            )
            # Predict based on X
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
        return_runner: bool = False
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
        
        # Check if batch size is defined or not
        if data_params.get("batch_size") is None:
            # Fall back to product of max cells and max classes per batch
            msb = data_params.get("max_cells_per_batch", 32)
            mcb = data_params.get("max_classes_per_batch", 16)
            batch_size = int(msb * mcb)
            data_params["batch_size"] = batch_size

        # Create data splitter
        data_splitter = ContrastiveDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            **data_params,
        )

        plan_kwargs = train_params.pop('plan_kwargs', {})
        # Add external labels to training if included during setup
        if REGISTRY_KEYS.CLS_EMB_KEY in self.adata_manager.data_registry:
            cls_emb_registry = self.adata_manager.data_registry[REGISTRY_KEYS.CLS_EMB_KEY]
            # Add external embedding to plan parameters
            plan_kwargs[REGISTRY_KEYS.CLS_EMB_KEY] = get_anndata_attribute(
                self.adata,
                cls_emb_registry.attr_name,
                cls_emb_registry.attr_key,
            )
        # Use contrastive loss in validation if that set uses the same splitter
        if data_params.get('use_contrastive_loader', None) in ['val', 'both']:
            plan_kwargs['use_contr_in_val'] = True
        # Share code to label mapping with training plan
        plan_kwargs['_code_to_label'] = self._code_to_label.copy()
        # create training plan
        training_plan = self._training_plan_cls(module=self.module, n_classes=self.n_labels, **plan_kwargs)
        check_val_every_n_epoch = train_params.pop('check_val_every_n_epoch', 1)
     
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
            # Don't log the class embedding
            plan_kwargs.pop(REGISTRY_KEYS.CLS_EMB_KEY, None)
            # save hyper-parameters to lightning logs
            hparams = {
                'data_params': data_params,
                'model_params': model_params,
                'plan_params': plan_kwargs,
                'train_params': train_params
            }
            runner.trainer.logger.log_hyperparams(hparams)
        # Train model
        if return_runner:
            return runner
        else:
            return runner()
        
    def _validate_anndata(
        self, adata: AnnOrMuData | None = None, copy_if_view: bool = True, extend_categories: bool = True
    ) -> AnnData:
        """Validate anndata has been properly registered, transfer if necessary."""
        if adata is None:
            adata = self.adata

        _check_if_view(adata, copy_if_view=copy_if_view)

        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            logger.info(
                "Input AnnData not setup with scvi-tools. "
                + "attempting to transfer AnnData setup"
            )
            self._register_manager_for_instance(self.adata_manager.transfer_fields(adata, extend_categories=extend_categories))
        else:
            # Case where correct AnnDataManager is found, replay registration as necessary.
            adata_manager.validate()
        return adata

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        unlabeled_category: str,
        layer: str | None = None,
        class_emb_uns_key: str | None = 'cls_embedding',
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        cast_to_csr: bool = True,
        raw_counts: bool = True,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Setup AnnData object for training a GPVI model.
        """
        if not isinstance(adata.X, sp.csr_matrix) and cast_to_csr:
            logging.info('Converting adata.X to csr matrix to boost training efficiency.')
            adata.X = sp.csr_matrix(adata.X)
        setup_method_args = cls._get_setup_method_args(**locals())
        
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=raw_counts),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # Add class embedding matrix with shape (n_labels, class_emb_dim)
        if class_emb_uns_key is not None and class_emb_uns_key in adata.uns:
            anndata_fields.append(
                StringUnsField(REGISTRY_KEYS.CLS_EMB_KEY, class_emb_uns_key)
            )
        # Create AnnData manager
        adata_manager = EmbAnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
