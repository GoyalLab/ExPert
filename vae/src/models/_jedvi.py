
from copy import deepcopy
from src.utils.common import to_tensor
from src.utils.preprocess import scale_1d_array
from src.modules._jedvae import JEDVAE
from src.modules._splitter import ContrastiveDataSplitter
from src.utils.constants import REGISTRY_KEYS
from src._train.plan import ContrastiveSupervisedTrainingPlan
from src.data._manager import EmbAnnDataManager

import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn.functional as F
import warnings
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight


from collections.abc import Sequence
from typing import Literal, Any, Iterator

from anndata import AnnData

# scvi imports
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner
from scvi.data._utils import get_anndata_attribute, _check_if_view
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
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
    VarmField,
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
    _LATENT_QZM_KEY = "jedvi_latent_qzm"
    _LATENT_QZV_KEY = "jedvi_latent_qzv"


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
        cls_weight_method: str | None = None,
        **model_kwargs,
    ):
        super().__init__(adata)
        self._model_kwargs = dict(model_kwargs)
        self.use_gene_emb = self.adata_manager.registry.get('field_registries', {}).get(REGISTRY_KEYS.GENE_EMB_KEY, None) is not None
        # Initialize indices and labels for this VAE
        self._setup()

        # Set number of classes
        n_labels = self.summary_stats.n_labels
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        # Add classification weights based on support in data set
        cls_weights = self._get_cls_weights(cls_weight_method)
        # Determine number of batches/datasets and fix library method
        n_batch = self.summary_stats.n_batch
        # Check number of hidden neurons, set to input features if < 0 else use given number
        n_hidden = self.summary_stats.n_vars if n_hidden < 0 else n_hidden

        # Initialize genevae
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=n_labels,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate_encoder=dropout_rate,
            dropout_rate_decoder=dropout_rate_decoder,
            class_embed_dim=self.n_dims_emb,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            linear_classifier=linear_classifier,
            cls_weights=cls_weights,
            **self._model_kwargs,
        )

        self.supervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.n_labels = n_labels
        self.use_full_cls_emb = False
        self.use_ctrl_emb = False
        # Give model summary
        self.n_unseen_labels = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['n_unseen_labels'] if REGISTRY_KEYS.CLS_EMB_INIT in adata.uns else None
        self._model_summary_string = (
            f"{self.__class__} Model with the following params: \n"
            f"n_classes: {self.n_labels}, "
            f"n_unseen_classes: {self.n_unseen_labels}, "
            f"use_gene_emb: {self.use_gene_emb} "
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
            class_weights = compute_class_weight(method, classes=self._label_mapping, y=labels)
            return torch.tensor(class_weights, dtype=torch.float32)

    def _setup(self):
        """Setup adata for training. Prepare embeddings"""
        # Initialize both embeddings as None
        self.gene_emb, self.cls_emb = None, None
        # Assign gene embedding
        if self.use_gene_emb:
            # Get gene embedding from adata
            gene_emb = self.adata.varm[REGISTRY_KEYS.GENE_EMB_KEY]
            # Add embedding dimensionality to encoder kwargs
            ext_en_kw = self._model_kwargs.get('extra_encoder_kwargs', {})
            emb_dim = gene_emb.shape[1]
            ext_en_kw['use_ext_emb'] = True
            ext_en_kw['gene_emb_dim'] = emb_dim
            self._model_kwargs['extra_encoder_kwargs'] = ext_en_kw
            logging.info(f'Initialized gene embedding with {emb_dim} dimensions')
            # Convert to dense if sparse
            if sp.issparse(gene_emb):
                gene_emb = gene_emb.todense()
            # Convert to tensor
            self.gene_emb = torch.Tensor(gene_emb).T

        # Assign class embedding
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self._label_mapping = labels_state_registry.categorical_mapping
        self.n_unseen_labels = None
        self._code_to_label = dict(enumerate(self._label_mapping))
        # order class embedding according to label mapping and transform to matrix
        if REGISTRY_KEYS.CLS_EMB_KEY in self.adata.uns:
            if REGISTRY_KEYS.CLS_EMB_INIT not in self.adata.uns:
                cls_emb: pd.DataFrame = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY]
                if not isinstance(cls_emb, pd.DataFrame):
                    logging.warning(f'Class embedding has to be a dataframe with labels as index, got {cls_emb.__class__}. Falling back to internal embedding.')
                else:
                    # Order external embedding to match label mapping and convert to csr matrix
                    _label_series = pd.Series(self._code_to_label.values())
                    _label_overlap = _label_series.isin(cls_emb.index)
                    _shared_labels = _label_series[_label_overlap].values
                    _unseen_labels = cls_emb.index.difference(_shared_labels)
                    self.n_train_labels = _shared_labels.shape[0]
                    self.n_unseen_labels = _unseen_labels.shape[0]
                    n_missing = _label_overlap.shape[0] - _label_overlap.sum()
                    if n_missing > 0:
                        raise ValueError(f'Found {n_missing} missing labels in cls embedding: {_label_series[~_label_overlap]}')

                    # Re-order embedding: first training labels then rest
                    cls_emb = pd.concat([cls_emb.loc[_shared_labels], cls_emb.loc[_unseen_labels]], axis=0)
                    # Include class similarity as pre-calculated matrix
                    logging.info(f'Calculating class similarities')
                    # Set gene embedding as class parameter
                    self.cls_emb = torch.tensor(cls_emb.values, dtype=torch.float32)
                    # Normalize embedding before calculating similarities
                    self.cls_emb = F.normalize(self.cls_emb, p=2, dim=-1)
                    # Set class similarities as class parameter
                    self.cls_sim = self.cls_emb @ self.cls_emb.T
                    # Save in adata for caching
                    self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY] = self.cls_emb
                    self.adata.uns[REGISTRY_KEYS.CLS_SIM_KEY] = self.cls_sim
                    # Save train embeddings and similarities as class parameters
                    self.train_cls_emb = self.cls_emb[:self.n_train_labels,:]
                    self.train_cls_sim = self.train_cls_emb @ self.train_cls_emb.T
                    # Get full list of perturbations for embedding
                    self.idx_to_label = cls_emb.index
                    # Save registration with this model
                    self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT] = {
                        'model': self.__class__,
                        'labels': cls_emb.index,
                        'n_train_labels': self.n_train_labels, 
                        'n_unseen_labels': self.n_unseen_labels,
                    }
            else:
                logging.info(f'Class embedding has already been initialized with {self.__class__} for this adata.')
                # Set to embeddings found in adata
                self.cls_emb = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY]
                self.cls_sim = self.adata.uns[REGISTRY_KEYS.CLS_SIM_KEY]
                # Init train embeddings from adata
                n_train = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['n_train_labels']
                self.train_cls_emb = self.cls_emb[:n_train,:]
                self.train_cls_sim = self.cls_sim[:n_train,:n_train]
                # Set indices
                self.idx_to_label = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['labels']
            # Set embedding dimension
            self.n_dims_emb = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY].shape[1]
        else:
            logging.info(f'No class embedding found in adata, falling back to internal embeddings with dimension 128. You can change this by specifying `n_dims_emb`.')
            self.n_dims_emb = self._model_kwargs.get('n_dims_emb', 128)

    def get_cls_emb(self, use_full_cls_emb: bool | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Helper getter to return either training or full class embedding"""
        if self.cls_emb is None:
            return None, None
        use_full_cls_emb = use_full_cls_emb if use_full_cls_emb is not None else self.use_full_cls_emb
        if use_full_cls_emb:
            return self.cls_emb, self.cls_sim
        else:
            return self.train_cls_emb, self.train_cls_sim

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **model_kwargs,
    ):
        """Initialize jedVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

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
            **scvi_setup_args,
        )
        model = cls(adata, **non_kwargs, **kwargs, **model_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        model.module.load_state_dict(scvi_state_dict, strict=False)
        model.was_pretrained = True

        return model
    
    @classmethod
    def from_base_model(
        cls,
        scvi_model,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        **model_kwargs,
    ):
        """Initialize jedVI model with weights from pretrained :class:`~scvi.model.JEDVI` model.

        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the JEDVI model. If None, uses the `labels_key` used to
            setup the JEDVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~JEDVI.setup_anndata`.
        model_kwargs
            kwargs for JEDVI model
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
    
    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5_000,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Compute the latent representation of the data.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            :class:`~anndata.AnnData` object with :attr:`~anndata.AnnData.var_names` in the same
            order as the ones used to train the model. If ``None`` and ``dataloader`` is also
            ``None``, it defaults to the object used to initialize the model.
        indices
            Indices of observations in ``adata`` to use. If ``None``, defaults to all observations.
            Ignored if ``dataloader`` is not ``None``
        give_mean
            If ``True``, returns the mean of the latent distribution. If ``False``, returns an
            estimate of the mean using ``mc_samples`` Monte Carlo samples.
        mc_samples
            Number of Monte Carlo samples to use for the estimator for distributions with no
            closed-form mean (e.g., the logistic normal distribution). Not used if ``give_mean`` is
            ``True`` or if ``return_dist`` is ``True``.
        batch_size
            Minibatch size for the forward pass. If ``None``, defaults to
            ``scvi.settings.batch_size``. Ignored if ``dataloader`` is not ``None``
        return_dist
            If ``True``, returns the mean and variance of the latent distribution. Otherwise,
            returns the mean of the latent distribution.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.

        Returns
        -------
        An array of shape ``(n_obs, n_latent)`` if ``return_dist`` is ``False``. Otherwise, returns
        a tuple of arrays ``(n_obs, n_latent)`` with the mean and variance of the latent
        distribution.
        """
        from torch.distributions import Normal
        from torch.nn.functional import softmax

        from scvi.module._constants import MODULE_KEYS

        self._check_if_trained(warn=False)
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        if dataloader is None:
            adata = self._validate_anndata(adata, extend_categories=True)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

        zs: list[Tensor] = []
        qz_means: list[Tensor] = []
        qz_vars: list[Tensor] = []
        for tensors in dataloader:
            tensors[REGISTRY_KEYS.CLS_EMB_KEY], _ = self.get_cls_emb()
            tensors[REGISTRY_KEYS.GENE_EMB_KEY] = self.gene_emb
            outputs: dict[str, Tensor | Distribution | None] = self.module.inference(
                **self.module._get_inference_input(tensors)
            )

            if MODULE_KEYS.QZ_KEY in outputs:
                qz: Distribution = outputs.get(MODULE_KEYS.QZ_KEY)
                qzm: Tensor = qz.loc
                qzv: Tensor = qz.scale.square()
            else:
                qzm: Tensor = outputs.get(MODULE_KEYS.QZM_KEY)
                qzv: Tensor = outputs.get(MODULE_KEYS.QZV_KEY)
                qz: Distribution = Normal(qzm, qzv.sqrt())

            if return_dist:
                qz_means.append(qzm.cpu())
                qz_vars.append(qzv.cpu())
                continue

            z: Tensor = qzm if give_mean else outputs.get(MODULE_KEYS.Z_KEY)

            if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                samples = qz.sample([mc_samples])
                z = softmax(samples, dim=-1).mean(dim=0)

            zs.append(z.cpu())

        if return_dist:
            return torch.cat(qz_means).numpy(), torch.cat(qz_vars).numpy()
        else:
            return torch.cat(zs).numpy()

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        soft: bool = False,
        batch_size: int | None = None,
        return_latent: bool = True,
        use_full_cls_emb: bool | None = None,
        **kwargs
    ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray] | np.ndarray | pd.DataFrame:
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~JEDVI.setup_anndata`.
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
        # Log usage of gene embeddings
        if self.gene_emb is not None:
            logging.info(f'Using model gene embeddings ({self.gene_emb.shape})')
        # validate adata or get it from model
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )
        # Get class embeddings from model
        cls_emb, _, = self.get_cls_emb(use_full_cls_emb=use_full_cls_emb)
        if cls_emb is not None:
            logging.info(f'Using class embedding: {cls_emb.shape}')
        
        y_pred = []
        y_cz = []
        for _, tensors in enumerate(scdl):
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            pred, cz = self.module.classify(
                x,
                batch_index=batch,
                g=self.gene_emb,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
                class_embeds=cls_emb
            )
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())
            y_cz.append(cz.detach().cpu())
        # Concatenate batch results
        y_pred = torch.cat(y_pred).numpy()
        y_cz = torch.cat(y_cz).numpy()
        if not soft:
            predictions = []
            for p in y_pred:
                if self.cls_emb is None:
                    label = self._code_to_label[p]
                else:
                    label = self.idx_to_label[p]
                predictions.append(label)

            pred = np.array(predictions)
        else:
            n_labels = len(pred[0])
            pred = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels] if self.cls_emb is None else self.idx_to_label[:n_labels],
                index=adata.obs_names[indices],
            )
        if return_latent:
            return pred, y_cz
        else:
            return pred
        
    def get_training_plan(self, **plan_kwargs):
        cls_emb, cls_sim = self.get_cls_emb()
        return self._training_plan_cls(
            module=self.module, 
            n_classes=self.n_labels, 
            cls_emb=cls_emb,
            cls_sim=cls_sim,
            gene_emb=self.gene_emb,
            **plan_kwargs
        )
        
    @devices_dsp.dedent
    def train(
        self,
        data_params: dict[str, Any]={}, 
        model_params: dict[str, Any]={}, 
        train_params: dict[str, Any]={},
        return_runner: bool = False,
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
        # Setup training plan
        plan_kwargs: dict = train_params.pop('plan_kwargs', {})

        # Use contrastive loss in validation if that set uses the same splitter
        if data_params.get('use_contrastive_loader', None) in ['val', 'both']:
            plan_kwargs['use_contr_in_val'] = True
        # Share code to label mapping with training plan
        plan_kwargs['_code_to_label'] = self._code_to_label.copy()
        # create training plan
        training_plan = self.get_training_plan(**plan_kwargs)
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
        # Save training plan to model
        self.training_plan = training_plan
        # Update model summary to include if it's trained on fixed or full class embedding
        self.use_full_cls_emb = bool(plan_kwargs.get('use_full_cls_emb'))
        self._model_summary_string += f", use_full_cls_emb: {self.use_full_cls_emb}"
        self.__repr__()
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
    
    def create_test_data(self, adata: AnnData) -> AnnData:
        import anndata as ad
        # Subset test data features to the features the model has been trained on
        model_genes = self.adata.var_names
        v_mask = adata.var_names.isin(model_genes)
        adata._inplace_subset_var(v_mask)
        # Pad missing features with 0 vectors
        missing_genes = model_genes.difference(adata.var_names)
        n_missing = len(missing_genes)
        if n_missing > 0:
            logging.info(f'Found {n_missing} missing features in test data. Setting them to 0.')
            # Copy .var info from model's adata
            missing_var = self.adata[:,missing_genes].var
            # Retain cell-specfic information
            missing_obs = adata.obs.copy()
            # Create empty expression for missing genes for all cells
            missing_X = np.zeros((adata.shape[0], missing_genes.shape[0]))
            missing_X = sp.csr_matrix(missing_X)
            # Create padded adata object
            missing_adata = AnnData(X=missing_X, obs=missing_obs, var=missing_var)
            uns = deepcopy(adata.uns)
            adata = ad.concat([adata, missing_adata], axis=1)
            adata.obs = missing_obs
            adata.uns = uns
        # Sort .var to same order as observed in model's adata
        return adata[:,self.adata.var_names].copy()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        layer: str | None = None,
        class_emb_uns_key: str | None = 'cls_embedding',
        class_certainty_key: str | None = None,
        gene_emb_varm_key: str | None = None,
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
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # Add class embedding matrix with shape (n_labels, class_emb_dim)
        if class_emb_uns_key is not None and class_emb_uns_key in adata.uns:
            logging.info(f'Adding class embedding from adata.uns[{class_emb_uns_key}] to model')
            anndata_fields.append(StringUnsField(REGISTRY_KEYS.CLS_EMB_KEY, class_emb_uns_key))
        # Add gene embedding matrix with shape (n_vars, emb_dim)
        if gene_emb_varm_key is not None and gene_emb_varm_key in adata.varm:
            logging.info(f'Adding gene embedding from adata.varm[{gene_emb_varm_key}] to model')
            anndata_fields.append(VarmField(REGISTRY_KEYS.GENE_EMB_KEY, gene_emb_varm_key))
        # Add class score per cell, ensure its scaled [0, 1]
        if class_certainty_key is not None:
            logging.info(f'Adding class certainty score from adata.obs[{class_certainty_key}] to model')
            scaled_class_cert_key = f'{class_certainty_key}_scaled'
            adata.obs[scaled_class_cert_key] = scale_1d_array(adata.obs[class_certainty_key].astype(float).values)
            anndata_fields.append(NumericalObsField(REGISTRY_KEYS.CLS_CERT_KEY, scaled_class_cert_key, required=False))
        # Create AnnData manager
        adata_manager = EmbAnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
