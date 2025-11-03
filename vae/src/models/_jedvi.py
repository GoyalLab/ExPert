import os
from src.utils.constants import REGISTRY_KEYS, EXT_CLS_EMB_INIT, PREDICTION_KEYS, MODULE_KEYS
import src.utils.io as io
import src.utils.performance as pf
from src.utils.preprocess import scale_1d_array
from src.utils._callbacks import DelayedEarlyStopping
import src.utils.plotting as pl
from src.modules._jedvae import JEDVAE
from src.modules._splitter import ContrastiveDataSplitter
from src._train.plan import ContrastiveSupervisedTrainingPlan
from src.data._manager import EmbAnnDataManager

from copy import deepcopy
import torch
from torch import Tensor
from torch.distributions import Distribution
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight


from collections.abc import Sequence
from collections import OrderedDict
from typing import Literal, Any, Iterator

from anndata import AnnData

# scvi imports
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner, SaveCheckpoint
from scvi.data._utils import _get_adata_minify_type, _check_if_view, _is_minified
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
from scvi.model.base import (
    ArchesMixin,
    VAEMixin,
    RNASeqMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass
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

import logging
log = logging.getLogger(__name__)


class JEDVI(
        RNASeqMixin, 
        VAEMixin,
        ArchesMixin, 
        UnsupervisedTrainingMixin,
        BaseMinifiedModeModelClass
    ):
    _module_cls = JEDVAE
    _name = __qualname__
    _training_plan_cls = ContrastiveSupervisedTrainingPlan
    _LATENT_QZM_KEY = f"{__qualname__}_latent_qzm"
    _LATENT_QZV_KEY = f"{__qualname__}_latent_qzv"
    _LATENT_ZC_KEY = f"{__qualname__}_latent_zc"
    _LATENT_CW_KEY = f"{__qualname__}_latent_cw"
    _LATENT_Z2C_KEY = f"{__qualname__}_latent_z2c"
    _LATENT_C2Z_KEY = f"{__qualname__}_latent_c2z"
    _LATENT_UMAP = f"{__qualname__}_latent_umap"


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
        ctrl_class: str | None = None,
        linear_classifier: bool = False,
        cls_weight_method: str | None = None,
        use_full_cls_emb: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        self._model_kwargs = dict(model_kwargs)
        self.use_gene_emb = self.adata_manager.registry.get('field_registries', {}).get(REGISTRY_KEYS.GENE_EMB_KEY, None) is not None
        self.ctrl_class = ctrl_class
        # Set number of classes
        self.n_labels = self.summary_stats.n_labels
        self.n_batches = self.summary_stats.n_batch
        # Whether to use the full class embedding
        self.use_full_cls_emb = use_full_cls_emb
        # Create placeholder for log directory
        self.model_log_dir = None

        # Initialize indices and labels for this VAE
        self._setup()
        # Get covariates
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
        # Get class embeddings
        cls_emb, cls_sim = self.get_cls_emb()
        # Initialize genevae
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.n_labels,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            cls_emb=cls_emb,
            cls_sim=cls_sim,
            ctx_emb=self.ctx_emb,
            dropout_rate_encoder=dropout_rate,
            dropout_rate_decoder=dropout_rate_decoder,
            ext_class_embed_dim=self.ext_class_embed_dim,
            ctrl_class_idx=self.ctrl_class_idx,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            linear_classifier=linear_classifier,
            cls_weights=cls_weights,
            **self._model_kwargs,
        )
        # Fresh initialization, set params accordingly
        self.supervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.is_evaluated = False
        self.use_ctrl_emb = False
        # Give model summary
        self.n_unseen_labels = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['n_unseen_labels'] if REGISTRY_KEYS.CLS_EMB_INIT in adata.uns else None
        self._model_summary_string = (
            f"{self.__class__} Model with the following params: \n"
            f"n_classes: {self.n_labels}, "
            f"n_unseen_classes: {self.n_unseen_labels}, \n"
            f"n_contexts: {self.n_batches}, "
            f"n_unseen_contexts: {self.n_unobs_ctx}, \n" if self.n_unobs_ctx > 0 else "\n"
            f"use_gene_emb: {self.use_gene_emb}"
        )
        # Include control class information
        if self.ctrl_class is not None and self.ctrl_class_idx is not None:
            self._model_summary_string += f"\nctrl_class: {self.ctrl_class}"
            self._model_summary_string += f"\nuse_learnable_control_emb: {self.module.use_learnable_control_emb}"

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
            log.info(f'Initialized gene embedding with {emb_dim} dimensions')
            # Convert to dense if sparse
            if sp.issparse(gene_emb):
                gene_emb = gene_emb.todense()
            # Convert to tensor
            self.gene_emb = torch.Tensor(gene_emb).T

        # Assign batch labels
        batch_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        self.original_batch_key = batch_state_registry.original_key
        self._batch_mapping = batch_state_registry.categorical_mapping
        self.idx_to_batch = pd.Series(self._batch_mapping)
        # Add context embedding if given
        ctx_emb_registry = self.adata_manager.registry['field_registries'].get(REGISTRY_KEYS.CTX_EMB_KEY, {})
        self.ctx_emb = None
        self.n_unobs_ctx = 0
        if len(ctx_emb_registry) > 0:
            # Get registry keys for context embedding in adata
            ext_ctx_emb_key = ctx_emb_registry.get('original_key')
            ctx_emb_key = ctx_emb_registry['data_registry'].get('attr_key') if ext_ctx_emb_key is None else ext_ctx_emb_key
            # Get the embedding from adata and check if it's a dataframe
            if not isinstance(self.adata.uns[ctx_emb_key], pd.DataFrame):
                log.warning(f'Context embedding has to be a dataframe with contexs as index, got {ctx_emb.__class__}. Falling back to internal embedding.')
            else:
                # Extract context embedding from adata
                ctx_emb: pd.DataFrame = self.adata.uns[ctx_emb_key]
                # Get observed and unobserved context labels
                _obs_contexts_mask = self.idx_to_batch.isin(ctx_emb.index)
                if _obs_contexts_mask.sum() != self.idx_to_batch.shape[0]:
                    raise ValueError(f'Missing contexts in external context embedding. {self.idx_to_batch[~_obs_contexts_mask].values}')
                _obs_ctx_labels = self.idx_to_batch[_obs_contexts_mask]
                _unobs_ctx_labels = ctx_emb.index.difference(_obs_ctx_labels)
                self.n_unobs_ctx = _unobs_ctx_labels.shape[0]
                # Order embedding by observed contexts and unobserved after and update class variable
                ctx_emb = pd.concat([ctx_emb.loc[_obs_ctx_labels], ctx_emb.loc[_unobs_ctx_labels]], axis=0)
                # Convert to tensor
                self.ctx_emb = io.to_tensor(ctx_emb)
            
        # Assign class embedding
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self._label_mapping = labels_state_registry.categorical_mapping
        self.n_unseen_labels = None
        self._code_to_label = dict(enumerate(self._label_mapping))
        # Check if adata has an external class embedding registered
        cls_emb_registry = self.adata_manager.registry['field_registries'].get(REGISTRY_KEYS.CLS_EMB_KEY, {})
        if len(cls_emb_registry) > 0:
            # Get adata.uns key for embedding, fall back to attribute key if no original key is given
            ext_cls_emb_key = cls_emb_registry.get('original_key')
            cls_emb_key = cls_emb_registry['data_registry'].get('attr_key') if ext_cls_emb_key is None else ext_cls_emb_key
            # Check if adata has already been registered with this model class
            if REGISTRY_KEYS.CLS_EMB_INIT in self.adata.uns and self._name == self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.MODEL_KEY]:
                log.info(f'Adata has already been initialized with {self.__class__}, loading model settings from adata.')
                # Set to embeddings found in adata
                self.cls_emb = io.to_tensor(self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY])
                self.cls_sim = io.to_tensor(self.adata.uns[REGISTRY_KEYS.CLS_SIM_KEY])
                # Init train embeddings from adata
                self.n_train_labels = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.N_TRAIN_LABELS_KEY]
                self.train_cls_emb = self.cls_emb[:self.n_train_labels,:]
                self.train_cls_sim = self.cls_sim[:self.n_train_labels,:self.n_train_labels]
                # Set indices
                self.idx_to_label = np.array(self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.LABELS_KEY])
                # Set control class index
                self.ctrl_class = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.CTRL_CLASS_KEY]
                self.ctrl_class_idx = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.CTRL_CLASS_IDX_KEY]
            else:
                # Register adata's class embedding with this model
                cls_emb: pd.DataFrame = self.adata.uns[cls_emb_key]
                if not isinstance(cls_emb, pd.DataFrame):
                    log.warning(f'Class embedding has to be a dataframe with labels as index, got {cls_emb.__class__}. Falling back to internal embedding.')
                else:
                    # Order external embedding to match label mapping and convert to csr matrix
                    _label_series = pd.Series(self._code_to_label.values())
                    _label_overlap = _label_series.isin(cls_emb.index)
                    _shared_labels = _label_series[_label_overlap].values
                    _unseen_labels = cls_emb.index.difference(_shared_labels)
                    # Remove control embedding if given
                    ctrl_class_idx_matches = np.where(_label_series==self.ctrl_class)[0]
                    ctrl_exists_in_data = len(ctrl_class_idx_matches) > 0
                    if not ctrl_exists_in_data:
                        log.warning(f'Specified control label {self.ctrl_class} is not in adata class labels, ignoring parameter.')
                    if self.ctrl_class is not None and ctrl_exists_in_data:
                        # Find control index
                        self.ctrl_class_idx = ctrl_class_idx_matches[0]
                        # Create empty embedding for control
                        if self.ctrl_class not in _shared_labels:
                            log.info(f'Adding empty control external class embedding, will be learned by model.')
                            # Create empty embedding at last slot
                            dummy_emb = pd.DataFrame(np.zeros((1, cls_emb.shape[-1])), index=[self.ctrl_class], columns=cls_emb.columns)
                            # Add dummy embedding as 
                            _shared_cls_emb = cls_emb.loc[_shared_labels]
                            _unseen_cls_emb = cls_emb.loc[_unseen_labels]
                            cls_emb = pd.concat((_shared_cls_emb, dummy_emb, _unseen_cls_emb), axis=0)
                            # Set control index to first embedding index
                            _shared_labels = np.concatenate((
                                np.array(_shared_labels), np.array([self.ctrl_class])
                            ))
                        else:
                            log.info(f'Overwriting existing external control class embedding, will be learned by model.')
                        # Reset label overlap
                        _label_overlap = _label_series.isin(_shared_labels)
                    else:
                        # Set no label as control class
                        self.ctrl_class_idx = None
                    self.n_train_labels = _shared_labels.shape[0]
                    self.n_unseen_labels = _unseen_labels.shape[0]
                    n_missing = _label_overlap.shape[0] - _label_overlap.sum()
                    if n_missing > 0:
                        raise ValueError(f'Found {n_missing} missing labels in external class embedding: {_label_series[~_label_overlap]}')
                    # Re-order embedding: first training labels then rest
                    cls_emb = pd.concat([cls_emb.loc[_shared_labels], cls_emb.loc[_unseen_labels]], axis=0)
                    # Include class similarity as pre-calculated matrix
                    log.info(f'Calculating external class similarities')
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
                    self.idx_to_label = cls_emb.index.values
                    # Save registration with this model
                    self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT] = {
                        EXT_CLS_EMB_INIT.MODEL_KEY: self._name,
                        EXT_CLS_EMB_INIT.LABELS_KEY: self.idx_to_label,
                        EXT_CLS_EMB_INIT.N_TRAIN_LABELS_KEY: self.n_train_labels, 
                        EXT_CLS_EMB_INIT.N_UNSEEN_LABELS_KEY: self.n_unseen_labels,
                        EXT_CLS_EMB_INIT.CTRL_CLASS_KEY: self.ctrl_class,
                        EXT_CLS_EMB_INIT.CTRL_CLASS_IDX_KEY: self.ctrl_class_idx,
                    }
            # Set embedding dimension
            self.ext_class_embed_dim = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY].shape[1]
            # Use class embedding classifier
            self.has_ext_emb = True
        else:
            self.idx_to_label = np.array(self._code_to_label.values())
            self.ext_class_embed_dim = None
            self.ctrl_class_idx = None
            # Use base classifier
            self.has_ext_emb = False

    def get_cls_emb(self, use_full_cls_emb: bool | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Helper getter to return either training or full class embedding"""
        if self.cls_emb is None:
            return None, None
        # Check if we want to train on full or observed embedding only
        use_full_cls_emb = use_full_cls_emb if use_full_cls_emb is not None else self.use_full_cls_emb
        if use_full_cls_emb:
            return self.cls_emb, self.cls_sim
        else:
            return self.train_cls_emb, self.train_cls_sim
    
    @classmethod
    def from_base_model(
        cls,
        pretrained_model,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        excl_setup_keys: list[str] = ['class_emb_uns_key'],
        excl_states: list[str] | None = None,
        freeze_modules: list[str] | None = ['z_encoder', 'decoder'],
        check_model_kwargs: bool = True,
        **model_kwargs,
    ):
        """Initialize jedVI model with weights from pretrained :class:`~src.models.JEDVI` model.

        Parameters
        ----------
        TODO: add parameter description
        """
        import re
        from scvi.data._constants import (
            _SETUP_ARGS_KEY,
        )

        pretrained_model._check_if_trained(message="Passed in model hasn't been trained yet.")

        model_kwargs = dict(model_kwargs)
        init_params = pretrained_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (_, j) in kwargs.items() for (k, v) in j.items()}
        # Collect all params for model init
        model_params = non_kwargs
        model_params.update(kwargs)
        # Check if model kwargs options diverged between pre-trained model and these configs
        if check_model_kwargs:
            for k, _ in {**non_kwargs, **kwargs}.items():
                if k in model_kwargs.keys():
                    log.warning(f'Duplicate model config key detected, removing: {k}')
                    del model_kwargs[k]
        model_params.update(model_kwargs)
        # Fall back to pre-training data if no new traiing data is provided
        if adata is None:
            adata = pretrained_model.adata
        else:
            if _is_minified(adata):
                raise ValueError(f"Please provide a non-minified `adata` to initialize {cls.__qualname__}.")
            # validate new anndata against old model
            pretrained_model._validate_anndata(adata)
        # Use pretraining adata setup kwargs
        pretrained_setup_args = deepcopy(pretrained_model.adata_manager.registry[_SETUP_ARGS_KEY])
        pretrained_labels_key = pretrained_setup_args["labels_key"]
        # Set labels for classification if not already specified in pre-training
        if labels_key is None and pretrained_labels_key is None:
            # Actually this can never be the case lol
            raise ValueError(
                "A `labels_key` is necessary because the pre-trained model did not have one."
            )
        if pretrained_labels_key is None:
            pretrained_setup_args.update({"labels_key": labels_key})
        # Exclude adata setup keys from pr-etraining
        pretrained_setup_args = {k: v for k, v in pretrained_setup_args.items() if k not in excl_setup_keys}
        # Setup new training adata
        cls.setup_anndata(
            adata,
            **pretrained_setup_args,
        )
        # Create fine-tune model
        model = cls(adata, **model_params)
        # Load pre-trained model weights
        pretrained_states = pd.Series(pretrained_model.module.state_dict().keys(), dtype=str)

        if excl_states is not None and len(excl_states) > 0:
            # Build a single regex pattern that matches any excluded state
            # (escapes special chars and joins with '|')
            pattern = "|".join(map(re.escape, excl_states))

            # Create a boolean mask: True = keep, False = exclude
            mask = ~pretrained_states.str.contains(pattern)

            # Filter state dict
            filtered_keys = pretrained_states[mask].tolist()
            pretrained_state_dict = OrderedDict({
                k: v for k, v in pretrained_model.module.state_dict().items() if k in filtered_keys
            })
        else:
            # Include all states
            pretrained_state_dict = pretrained_model.module.state_dict()
        # Load pre-trained weights
        model.module.load_state_dict(pretrained_state_dict, strict=False)
        # Label model as pre-trained
        model.was_pretrained = True
        # Transfer data splitter indices to new model
        model.train_indices = pretrained_model.train_indices
        model.validation_indices = pretrained_model.validation_indices
        model.test_indices = pretrained_model.test_indices
        # Freeze pre-trained models for next stage
        if freeze_modules is not None and isinstance(freeze_modules, list):
            for module in freeze_modules:
                model.module.freeze_module(module)
        # Return loaded pre-trained model
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

        # Check if model has been trained
        self._check_if_trained(warn=False)
        # Check if both adata and another dataloader are given
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        # Check if model is minified TODO: only works if using model's minified adata
        if self.is_minified and adata is None:
            qzm = self.adata.obsm[self._LATENT_QZM_KEY]
            qzv = self.adata.obsm[self._LATENT_QZV_KEY]
            qz: Distribution = Normal(qzm, qzv.sqrt())
            if return_dist:
                return qzm, qzv
            else:
                return qz

        # Adata is not minified
        if dataloader is None:
            adata = self._validate_anndata(adata, extend_categories=True)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

        zs: list[Tensor] = []
        qz_means: list[Tensor] = []
        qz_vars: list[Tensor] = []
        with torch.no_grad():
            for tensors in dataloader:
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
        batch_size: int | None = None,
        use_full_cls_emb: bool | None = None,
        use_posterior_mean: bool = True,
        **kwargs
    ) -> dict[str, pd.DataFrame | np.ndarray]:
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~JEDVI.setup_anndata`.
        indices
            Return probabilities for each class label.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        use_posterior_mean
            If ``True``, uses the mean of the posterior distribution to predict celltype
            labels. Otherwise, uses a sample from the posterior distribution - this
            means that the predictions will be stochastic.
        """
        # Check if model has been trained
        self._check_if_trained(warn=False)
        # Log usage of gene embeddings
        if self.gene_emb is not None:
            log.info(f'Using model gene embeddings ({self.gene_emb.shape})')
        # validate adata or get it from model
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size
        )
        # Get class embeddings from model
        if use_full_cls_emb is None or not use_full_cls_emb:
            cls_emb = self.module.class_embedding(device=self.device)
        # Fall back to full model embeddings
        else:
            cls_emb = self.cls_emb

        if cls_emb is not None:
            log.info(f'Using class embedding: {cls_emb.shape}')
        
        # Regular predictions
        y_pred = []
        # Arc-specific predictions
        zs = []
        Ws = []
        # Aligned predictions
        y_aligned_pred = []
        z2cs = []
        c2zs = []
        ls = []
        with torch.no_grad():
            for _, tensors in enumerate(scdl):
                # Unpack batch tensor
                x = tensors.get(REGISTRY_KEYS.X_KEY)
                l = tensors[REGISTRY_KEYS.LABELS_KEY].squeeze(-1)
                batch = tensors[REGISTRY_KEYS.BATCH_KEY]
                # Get continous covariates
                cont_key = REGISTRY_KEYS.CONT_COVS_KEY
                cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
                # Get categorical covariates
                cat_key = REGISTRY_KEYS.CAT_COVS_KEY
                cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
                # TODO: cachce inference if adata is minified
                if _is_minified(adata):
                    inference_outputs = {
                        MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                        MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY], 
                    }
                else:
                    inference_outputs = None
                # Run classification process
                cls_output = self.module.classify(
                    x,
                    batch_index=batch,
                    g=self.gene_emb,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    use_posterior_mean=use_posterior_mean,
                    labels=l,
                    inference_outputs=inference_outputs
                )
                # Handle output based on classifier used
                if self.module.has_classifier_latent():
                    # Unpack embedding projections
                    pred, z, W = cls_output
                    zs.append(z.detach().cpu())
                    Ws.append(W.detach().cpu())
                else:
                    # Set embeddings to z, TODO: set to None or empty tensors?
                    pred = cls_output        
                # Add batch predictions to overall predictions
                y_pred.append(pred.detach().cpu())

                # Get aligned predictions and latent spaces
                if self.module.has_align_cls:
                    aligned_logits, z2c, c2z = self.module.align(
                        x,
                        batch_index=batch,
                        g=self.gene_emb,
                        cat_covs=cat_covs,
                        cont_covs=cont_covs,
                        use_posterior_mean=use_posterior_mean,
                        class_embeds=cls_emb,
                    )
                    # Get aligned predictions
                    y_aligned_pred.append(aligned_logits.detach().cpu())
                    z2cs.append(z2c.detach().cpu())
                    c2zs.append(c2z.detach().cpu())
                # Log labels
                ls.append(l.detach().cpu())
        
        # Concatenate batch results
        ls = torch.cat(ls).numpy()
        y_pred = torch.cat(y_pred).numpy()
        # Get actual class labels for predictions
        n_labels = len(pred[0])
        soft_predictions = pd.DataFrame(
            y_pred,
            columns=self._label_mapping[:n_labels] if self.cls_emb is None else self.idx_to_label[:n_labels],
            index=adata.obs_names[indices],
        )
        # Take top prediction labels
        predictions = soft_predictions.columns[np.argmax(soft_predictions, axis=-1)]
        # Create base result object
        results = {
            PREDICTION_KEYS.PREDICTION_KEY: predictions,
            PREDICTION_KEYS.SOFT_PREDICTION_KEY: soft_predictions,
            'labels': ls,
        }
        # Check classifier latents
        if len(zs) > 0 and len(Ws) > 0:
            # Add classifier latents
            results.update({
                PREDICTION_KEYS.ZS_KEY: torch.cat(zs).numpy(),
                PREDICTION_KEYS.WS_KEY: torch.stack(Ws, dim=0).mean(dim=0).numpy()
            })
            
        # Check aligned latents
        if len(y_aligned_pred) > 0 and len(z2cs) > 0 and len(c2zs) > 0:
            # Add aligned latents
            y_aligned_pred = torch.cat(y_aligned_pred).numpy()
            # Get actual class labels for predictions
            n_algined_labels = len(aligned_logits[0])
            aligned_soft_predictions = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_algined_labels] if self.cls_emb is None else self.idx_to_label[:n_algined_labels],
                index=adata.obs_names[indices],
            )
            # Take top prediction labels
            aligned_predictions = aligned_soft_predictions.columns[np.argmax(aligned_soft_predictions, axis=-1)]
            results.update({
                PREDICTION_KEYS.ALIGN_PREDICTION_KEY: aligned_predictions,
                PREDICTION_KEYS.ALIGN_SOFT_PREDICTION_KEY: aligned_soft_predictions,
                PREDICTION_KEYS.ZS_KEY: torch.cat(z2cs).numpy(),
                PREDICTION_KEYS.WS_KEY: torch.stack(c2zs, dim=0).mean(dim=0).numpy()
            })
        # Return predictions
        return results
        
    def get_training_plan(self, **plan_kwargs):
        return self._training_plan_cls(
            module=self.module, 
            n_classes=self.n_labels, 
            gene_emb=self.gene_emb,
            **plan_kwargs
        )
    
    @property
    def is_minified(self) -> bool:
        return _get_adata_minify_type(self.adata) is not None
        
    @devices_dsp.dedent
    def train(
        self,
        config: dict[str, Any], 
        minify_adata: bool = False,
        cache_data_splitter: bool = True,
    ):
        """Train the model.

        Parameters
        ----------
        TODO: fill in parameteres
        data_params: dict
            **kwargs for src.modules._splitter.ContrastiveDataSplitter
        """
        from src.tune._statics import CONF_KEYS
        # Unpack run config dictionary
        data_params: dict[str, Any] = config.get(CONF_KEYS.DATA, {})
        model_params: dict[str, Any] = config.get(CONF_KEYS.MODEL, {})
        train_params: dict[str, Any] = config.get(CONF_KEYS.TRAIN, {})
        # Get max epochs specified in params
        epochs = train_params.get('max_epochs')
        # Determine number of epochs needed for complete training
        max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
        # Train for suggested number of epochs if no specification is give
        if epochs is None:
            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))
            epochs = max_epochs
        log.info(f'Epochs suggested: {max_epochs}, training for {epochs} epochs.')
        # Get training split fraction
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
        # Add control class index to data params
        data_params['ctrl_class'] = self.ctrl_class

        # Cache indices if model was pre-trained
        if self.was_pretrained and cache_data_splitter:
            log.info(f'Re-using pre-training data split.')
            cache_indices = {
                'train': self.train_indices,
                'val': self.validation_indices,
                'test': getattr(self, 'test_indices', np.array([]))
            }
            # Add to splitter params
            data_params['cache_indices'] = cache_indices

        # Create data splitter
        data_splitter = ContrastiveDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            **data_params,
        )
        # Setup training plan
        plan_kwargs: dict = train_params.pop('plan_kwargs', {})
        # Specify which splits use contrastive loading
        plan_kwargs['use_contrastive_loader'] = data_splitter.use_contrastive_loader

        # Use contrastive loss in validation if that set uses the same splitter
        if data_params.get('use_contrastive_loader', None) in ['val', 'both']:
            plan_kwargs['use_contr_in_val'] = True
        # Share code to label mapping with training plan
        plan_kwargs['_code_to_label'] = self._code_to_label.copy()
        # Share batch labels with training plan
        plan_kwargs['batch_labels'] = self._batch_mapping.copy()
        # Create training plan
        training_plan = self.get_training_plan(**plan_kwargs)
        # Check if tensorboard logger is given
        logger = train_params.get('logger')
        callbacks = []
        # Manage early stopping callback
        early_stopping_start_epoch = train_params.pop('early_stopping_start_epoch')
        if train_params.get('early_stopping', False) and early_stopping_start_epoch is not None:
            # Disable scvi internal early stopping
            train_params['early_stopping'] = False
            # Create custom delayed early stopping using the start epoch parameter
            early_stopping_callback = DelayedEarlyStopping(
                start_epoch=early_stopping_start_epoch,
                monitor=train_params.get('early_stopping_monitor', 'validation_loss'),
                min_delta=train_params.get('early_stopping_min_delta', 0.0),
                patience=train_params.get('early_stopping_patience', 0.0),
                mode=train_params.get('early_stopping_mode', 0.0),
            )
            # Add to callbacks
            callbacks.append(early_stopping_callback)
        # Manage checkpoint callback
        use_checkpoint = train_params.pop('checkpoint', False)
        checkpoint_monitor = train_params.pop('checkpoint_monitor', 'validation_loss')
        checkpoint_mode = train_params.pop('checkpoint_mode', 'max')
        # Save most recent log dir TODO: ensure model_log_dir is saved with model
        self.model_log_dir = logger.log_dir if logger is not None else None
        checkpoint_dir = None
        # Create checkpoint for max validation f1-score model
        if logger is not None and use_checkpoint:
            # Create checkpoint instance, save only best model
            checkpoint_dir = f"{self.model_log_dir}/checkpoints"
            checkpoint_callback = SaveCheckpoint(
                dirpath=checkpoint_dir,  # match logger directory
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                save_top_k=1,
            )
            log.info(f'Saving model checkpoints to: {checkpoint_dir}')
            # Add to list of callbacks
            callbacks.append(checkpoint_callback)
        # Add callbacks to Trainer kwargs
        train_params['callbacks'] = callbacks
        # Check val every epoch if we have callbacks registered
        if len(callbacks) > 0:
            train_params['check_val_every_n_epoch'] = 1
     
        # Create training runner
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            accelerator='auto',
            devices='auto',
            enable_checkpointing=use_checkpoint,
            default_root_dir=checkpoint_dir,
            **train_params
        )
        # Save hyperparameters to model output dir
        if logger is not None:
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
        if not self.was_pretrained:
            self._model_summary_string += f"use_full_cls_emb: {self.use_full_cls_emb}"
        self.__repr__()
        # Save runner in model
        self.last_runner = runner
        # Train model
        runner()
        # Add latent representation to model
        self._register_latent_variables()
        # Minify the model if enabled and adata is not already minified
        if minify_adata and not self.is_minified:
            log.info(f'Minifying adata with latent distribution')
            self.minify_adata(use_latent_qzm_key=self._LATENT_QZM_KEY, use_latent_qzv_key=self._LATENT_QZV_KEY)
        
    def _validate_anndata(
        self, adata: AnnOrMuData | None = None, copy_if_view: bool = True, extend_categories: bool = True
    ) -> AnnData:
        """Validate anndata has been properly registered, transfer if necessary."""
        if adata is None:
            adata = self.adata

        _check_if_view(adata, copy_if_view=copy_if_view)

        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            log.info(
                "Input AnnData not setup with scvi-tools. "
                + "attempting to transfer AnnData setup"
            )
            self._register_manager_for_instance(self.adata_manager.transfer_fields(adata, extend_categories=extend_categories))
        else:
            # Case where correct AnnDataManager is found, replay registration as necessary.
            adata_manager.validate()
        return adata
    
    def _get_available_splits(self) -> list[str]:
        """Get list of available data splits in the model.
        
        Returns
        -------
        list[str]
            List of available split names ('train', 'val', 'test')
        """
        return [
            split for split, indices in {
                'train': getattr(self, 'train_indices', None),
                'val': getattr(self, 'validation_indices', None), 
                'test': getattr(self, 'test_indices', None)
            }.items() if indices is not None and len(indices) > 0
        ]
    
    def _get_split_indices(self, split: str) -> np.ndarray:
        """Get model split indices by split key (train, val/validation, test)"""
        if split not in self._get_available_splits():
            raise ValueError(f'Model does not have indices for split: {split}, has to be one of {self._get_available_splits()}')
        if split == 'train':
            return self.train_indices
        if split in ['val', 'validation']:
            return self.validation_indices
        if split == 'test':
            return self.test_indices
        return None
        
    def _get_split_adata(self, split: str, ignore_ctrl: bool = True) -> ad.AnnData:
        """Get AnnData subset for a specific split and optionally filter control cells.

        Parameters
        ----------
        split : str
            Data split to get ('train', 'val', 'test')
        ignore_ctrl : bool
            Whether to filter out control cells from the split

        Returns
        -------
        ad.AnnData
            AnnData subset for the specified split
        """
        # Return subset of self.adata based on mode indices
        split_indices = self._get_split_indices(split)
        # Get data for this split
        adata = self.adata[split_indices]
        # Ignore control cells if option is given and control class exists
        if self.ctrl_class is not None and ignore_ctrl:
            # Filter out control cells
            adata = adata[adata.obs[self.original_label_key] != self.ctrl_class]
        return adata
    
    def _get_model_classification_report(self, split: str, ignore_ctrl: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.metrics import classification_report

        # Check if internal predictions are available
        if PREDICTION_KEYS.PREDICTION_KEY not in self.adata.obs:
            raise ValueError('Run self.evaluate() before calculating statistics.')
        # Get mode adata
        mode_adata = self._get_split_adata(split, ignore_ctrl=ignore_ctrl)
        # Generate a classification report for the relevant mode
        report = classification_report(
            mode_adata.obs[self.original_label_key],
            mode_adata.obs[PREDICTION_KEYS.PREDICTION_KEY],
            zero_division=np.nan,
            output_dict=True
        )
        # Format output accordingly
        report_df = pd.DataFrame(report).transpose()
        summary = report_df[report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        report_data = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        report_data['log_count'] = np.log(report_data['support'])
        # Add mode to results
        report_data[REGISTRY_KEYS.SPLIT_KEY] = split
        summary[REGISTRY_KEYS.SPLIT_KEY] = split
        return summary, report_data
    
    def _register_classification_report(self, force: bool = False, ignore_ctrl: bool = True) -> None:
        """Run classification report for all registered modes."""
        # Check if already given
        if self.is_evaluated and not force:
            return
        # Generate classification summaries and reports
        summaries, reports = [], []
        for split in self._get_available_splits():
            summary, report = self._get_model_classification_report(split, ignore_ctrl=ignore_ctrl)
            summaries.append(summary)
            reports.append(report)
        summaries = pd.concat(summaries, axis=0)
        reports = pd.concat(reports, axis=0)
        self.adata.uns[PREDICTION_KEYS.SUMMARY_KEY] = summaries
        self.adata.uns[PREDICTION_KEYS.REPORT_KEY] = reports

    def _register_top_n_predictions(self) -> None:
        """Calculate performance metrics for each data split and optionally also for each context"""
        top_n_predictions = pf.compute_top_n_predictions(
            adata=self.adata,
            split_key=REGISTRY_KEYS.SPLIT_KEY,
            context_key=self.original_batch_key,
            labels_key=self.original_label_key,
            ctrl_key=self.ctrl_class,
            predictions_key=PREDICTION_KEYS.SOFT_PREDICTION_KEY,
        )
        # Add data to model's adata
        self.adata.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY] = top_n_predictions

    def _get_top_n_predictions(self) -> pd.DataFrame:
        if PREDICTION_KEYS.TOP_N_PREDICTION_KEY not in self.adata.obsm:
            raise KeyError('No model evaluation top prediction metrics available. Run model.evaluate()')
        return self.adata.obsm[PREDICTION_KEYS.TOP_N_PREDICTION_KEY]
    
    def _register_latent_variables(self, force: bool = True) -> None:
        """Register latent space with self.adata.obsm"""
        # Check if it is already present, recalculate if forced
        if self._LATENT_QZM_KEY in self.adata.obsm and not force:
            return
        # Run inference and assign it to model adata
        qzm, qzv = self.get_latent_representation(return_dist=True)
        self.adata.obsm[self._LATENT_QZM_KEY] = qzm
        self.adata.obsm[self._LATENT_QZV_KEY] = qzv

    def _register_split_labels(self, force: bool = True) -> None:
        """Add split label to self.adata.obs"""
        # Check if these labels are already in adata and reassign if forced to
        if REGISTRY_KEYS.SPLIT_KEY in self.adata.obs and not force:
            return
        # Create array of split labels matching adata size
        split_labels = np.full(self.adata.n_obs, None, dtype=object)
        # Fill in split labels based on indices
        for split in self._get_available_splits():
            split_indices = self._get_split_indices(split)
            split_labels[split_indices] = split
        # Add split labels to adata.obs
        self.adata.obs[REGISTRY_KEYS.SPLIT_KEY] = split_labels
    
    def _register_model_predictions(self, force: bool = True) -> None:
        """Add detailed model classification outputs to internal self.adata object"""
        # Check if model has been trained
        self._check_if_trained(warn=False)
        # Predictions have not yet been made
        if PREDICTION_KEYS.PREDICTION_KEY in self.adata.obs and not force:
            return
        # Add split information to self.adata
        self._register_split_labels(force=force)
        # Run classifier forward pass, return soft predictions and latent spaces
        po = self.predict()
        # Save regular classification predictions to model adata
        self.adata.obs[PREDICTION_KEYS.PREDICTION_KEY] = po[PREDICTION_KEYS.PREDICTION_KEY]
        self.adata.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = po[PREDICTION_KEYS.SOFT_PREDICTION_KEY]          # (cells, classes)
        # Save aligned predictions and latent spaces to model
        aligned_pred = po.get(PREDICTION_KEYS.ALIGN_PREDICTION_KEY)
        if aligned_pred is not None:
            self.adata.obs[PREDICTION_KEYS.ALIGN_PREDICTION_KEY] = aligned_pred
        # Add aligned soft prediction
        aligned_soft_pred = po.get(PREDICTION_KEYS.ALIGN_SOFT_PREDICTION_KEY)
        if aligned_soft_pred is not None:
            self.adata.obsm[PREDICTION_KEYS.ALIGN_SOFT_PREDICTION_KEY] = aligned_soft_pred
        # Save aligned latent spaces
        z2c = po.get(PREDICTION_KEYS.Z2C_KEY)
        if z2c is not None:
            self.adata.obsm[self._LATENT_Z2C_KEY] = z2c     # z to class space projection latent (cells, shared dim)
        c2z = po.get(PREDICTION_KEYS.C2Z_KEY)
        if c2z is not None:
            self.adata.uns[self._LATENT_C2Z_KEY] = c2z      # class embedding space (classes, shared dim)
        # Calculate top N predictions over model data
        self._register_top_n_predictions()
        # Set evaluation as completed
        self.is_evaluated = True

    def evaluate(
            self,
            plot: bool = True,
            save_anndata: bool = False,
            output_dir: str | None = None,
            results_mode: Literal['return', 'save'] | None | list[str] = 'save',
            force: bool = True,
            ignore_ctrl: bool = True,
        ) -> dict[pd.DataFrame] | None:
        """Run model evaluation for all available data splits registered with the model."""
        if self.is_evaluated and not force:
            log.info('This model has already been evaluated, pass force=True to re-evaluate.')
            return
        # Refactor results mode to always be a list of options or None
        results_mode: list[str] | None = [results_mode] if results_mode is not None and not isinstance(results_mode, list) else None
        # Run full prediction for all registered data
        log.info('Running model predictions on all data splits.')
        self._register_model_predictions(force=force)
        # Run classification report for all data splits
        log.info('Generating reports.')
        self._register_classification_report(force=force, ignore_ctrl=ignore_ctrl)
        # Save model to tensorboard logger directory if registered
        if getattr(self, 'model_log_dir', None) is not None:
            base_output_dir = self.model_log_dir
            model_output_dir = os.path.join(self.model_log_dir, 'model')
        # Try to fall back to output_dir parameter and skip if not specified
        else:
            if output_dir is not None:
                base_output_dir = output_dir
                model_output_dir = os.path.join(output_dir, 'model')
            else:
                log.warning('Could not find model tensorboard log directory or a specified "output_dir", skipping model saving.')
                base_output_dir = None
                model_output_dir = None
        # Plot evalutation results if specified
        log.info('Plotting evaluation results.')
        if plot:
            self._plot_evalutation(output_dir=base_output_dir)
        # Save model only if we have an output directory
        if model_output_dir is not None:
            # Always save anndata if its minified else fall back to parameter
            save_anndata = True if self.is_minified else save_anndata
            save_ad_txt = 'with' if save_anndata else 'without'
            adata_txt = 'adata (minified)' if self.is_minified else 'adata'
            log.info(f'Saving model {save_ad_txt} {adata_txt} to: {model_output_dir}')
            self.save(dir_path=model_output_dir, save_anndata=save_anndata, overwrite=True)
        # Create evalutaion return object
        return_obj = None
        if results_mode is not None:
            # Collect evaluation result metrics
            eval_report = self.get_eval_report()
            eval_summary = self.get_eval_summary()
            # Return evaluation results as dictionary
            if 'return' in results_mode:
                return_obj = {
                    PREDICTION_KEYS.REPORT_KEY: eval_report,
                    PREDICTION_KEYS.SUMMARY_KEY: eval_summary,
                }
            # Save reports as seperate files in output directory
            if 'save' in results_mode:
                log.info(f'Saving evaluation metrics to: {base_output_dir}')
                sum_o = os.path.join(base_output_dir, f'{self._name}_eval_summary.csv')
                rep_o = os.path.join(base_output_dir, f'{self._name}_eval_report.csv')
                eval_summary.to_csv(sum_o)
                eval_report.to_csv(rep_o)
        log.info(f'Evaluation done.')
        # Return 
        return return_obj
    
    def get_eval_summary(self) -> pd.DataFrame | None:
        if PREDICTION_KEYS.SUMMARY_KEY not in self.adata.uns:
            log.warning('No model evaluation summary available.')
            return None
        return self.adata.uns[PREDICTION_KEYS.SUMMARY_KEY]
    
    def get_eval_report(self) -> pd.DataFrame | None:
        if PREDICTION_KEYS.REPORT_KEY not in self.adata.uns:
            log.warning('No model evaluation report available.')
            return None
        return self.adata.uns[PREDICTION_KEYS.REPORT_KEY]
    
    def _plot_eval_split(self, split: str, plt_dir: str) -> None:
        """Plot confusion matrix per split."""
        # Get split data without control cells
        adata = self._get_split_adata(split, ignore_ctrl=False)
        # Get actual and predicted class labels
        y = adata.obs[self.original_label_key].values.astype(str)
        y_hat = adata.obs[PREDICTION_KEYS.PREDICTION_KEY].values.astype(str)
        # Create output file path
        o = os.path.join(plt_dir, f'cm_{split}.png')
        pl.plot_confusion(y, y_hat, plt_file=o)
        # Plot mode umap using combined projection
        groups = [self.original_batch_key, self.original_label_key]
        for group in groups:
            # Plot a umap projection for every group
            o = os.path.join(plt_dir, f'umap_{split}_{group}.png')
            pl.plot_umap(adata, slot=self._LATENT_UMAP, hue=group, output_file=o)

    def _plot_eval_splits(self, plt_dir: str) -> None:
        """Generate all split-specific plots."""
        for split in self._get_available_splits():
            self._plot_eval_split(split, plt_dir=plt_dir)
    
    def _plot_evalutation(self, output_dir: str | None, metric: str = 'f1-score') -> None:
        """Save plots associated to model evaluation."""
        if output_dir is None:
            log.warning('Evalutaion output directory is not available. Skipping plots.')
            return
        # Set plotting directory and create if needed
        plt_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plt_dir, exist_ok=True)
        # Save latent space to model adata if not already calculated
        self._register_latent_variables(force=False)
        # Calculate UMAP for latent space based on latent mean
        pl.calc_umap(self.adata, rep=self._LATENT_QZM_KEY, slot_key=self._LATENT_UMAP)
        # Plot full UMAP colored by data split
        umap_split_o = os.path.join(plt_dir, f'full_umap_{REGISTRY_KEYS.SPLIT_KEY}.png')
        pl.plot_umap(self.adata, slot=self._LATENT_UMAP, hue=REGISTRY_KEYS.SPLIT_KEY, output_file=umap_split_o)
        # Plot full UMAP colored by batch label
        umap_batch_o = os.path.join(plt_dir, f'full_umap_{self.original_batch_key}.png')
        pl.plot_umap(self.adata, slot=self._LATENT_UMAP, hue=self.original_batch_key, output_file=umap_batch_o)
        # Plot full UMAP colored by batch label
        umap_label_o = os.path.join(plt_dir, f'full_umap_{self.original_label_key}.png')
        pl.plot_umap(self.adata, slot=self._LATENT_UMAP, hue=self.original_label_key, output_file=umap_label_o)
        # Plot performance - support correlations
        support_corr_o = os.path.join(plt_dir, f'support_correlations.png')
        pl.plot_performance_support_corr(self.adata.uns[PREDICTION_KEYS.REPORT_KEY], o=support_corr_o, hue=REGISTRY_KEYS.SPLIT_KEY)
        # Plot performance metric over top N predictions in all splits
        top_n_o = os.path.join(plt_dir, f'top_n_{metric}.png')
        pl.plot_top_n_performance(
            self.adata.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY],
            output_file=top_n_o,
            metric=metric,
            mean_split='val'
        )
        # Plot individual splits
        self._plot_eval_splits(plt_dir)
        return
    
    def align_model_features(self, adata: AnnData) -> AnnData:
        # Subset test data features to the features the model has been trained on
        model_genes = self.adata.var_names
        v_mask = adata.var_names.isin(model_genes)
        adata._inplace_subset_var(v_mask)
        # Pad missing features with 0 vectors
        missing_genes = model_genes.difference(adata.var_names)
        n_missing = len(missing_genes)
        if n_missing > 0:
            log.info(f'Found {n_missing} missing features in test data.')
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
    
    def minify_query(self, adata: ad.AnnData) -> None:
        """Minify query test data."""
        if self._LATENT_QZM_KEY not in adata.obsm or self._LATENT_QZV_KEY not in adata.obsm:
            raise KeyError(f'Query data does not have latent representation. Run JEDVI.test() first.')
        # Set .X to dense matrix of zeros
        adata.X = sp.csr_matrix(np.zeros(adata.X.shape))
    
    def test(
        self,
        test_adata_p: str,
        cls_label: str = 'cls_label',
        batch_label: str = 'dataset',
        output_dir: str | None = None,
        incl_unseen: bool = False,
        use_fixed_dataset_label: bool = True,
        plot: bool = True,
        results_mode: Literal['return', 'save'] | None | list[str] = 'save',
        min_cells_per_class: int = 10,
        top_n: int = 10,
        ctrl_key: str | None = None,
        minify: bool = False,
        seed: int = 42,
    ) -> None:
        """Function to evaluate model on unseen data."""
        # Refactor results mode to always be a list of options or None
        results_mode: list[str] | None = [results_mode] if results_mode is not None and not isinstance(results_mode, list) else None
        # TODO: all this only works for labelled data, inplement option to handle unlabelled data
        # TODO: actually pretty easy, just set prediction column to unknown for all cells duh
        # TODO: make this more modular
        # Read test adata and filter columns
        log.info(f'Testing model with: {test_adata_p}')
        test_ad = io.read_adata(test_adata_p, cls_label=cls_label, min_cells_per_class=min_cells_per_class)
        # Filter test adata for trained classes
        train_cls = set(self.get_training_classes())
        test_cls = set(test_ad.obs[cls_label].unique())
        available_cls = set(self.idx_to_label)
        # Look for shared classes between testing set and trained data
        shared_cls = test_cls.intersection(train_cls)
        unseen_cls = test_cls.difference(train_cls)
        test_cls = test_cls.intersection(available_cls)
        # Check if any labels overlap
        if len(test_cls) == 0:
            raise ValueError(f'No overlapping class labels between model and test set.')
        # Include unseen classes in testing or fall back to shared classes only
        filter_cls = test_cls if incl_unseen else shared_cls
        test_ad._inplace_subset_obs(test_ad.obs[cls_label].isin(filter_cls))
        log.info(f'Testing on {len(filter_cls)}/{len(test_cls)} classes.')
        # Use a fixed dataset label for testing
        ds_name = os.path.basename(test_adata_p).split('.h5ad')[0]
        # Save original batch keys in adata
        orig_batch_key = f'orig_{batch_label}'
        test_ad.obs[orig_batch_key] = test_ad.obs[batch_label].values
        # Set dataset labels for classification
        if not use_fixed_dataset_label:
            log.info('Randomly drawing dataset labels from training data.')
            training_datasets = self.adata.obs[self.original_batch_key].unique()
            np.random.seed(seed)
            ds_names = np.random.choice(training_datasets, test_ad.shape[0])
            test_ad.obs[batch_label] = pd.Categorical(ds_names)
        else:
            log.info(f'Added {ds_name} as dataset key.')
            test_ad.obs[batch_label] = ds_name
        # Register new testing data with model
        self.setup_anndata(test_ad, labels_key=cls_label, batch_key=batch_label)
        # Make sure test adata's features are consistent with model
        self.align_model_features(test_ad)
        # Get latent representation of model
        qzm, qzv = self.get_latent_representation(adata=test_ad, return_dist=True)
        test_ad.obsm[self._LATENT_QZM_KEY] = qzm
        test_ad.obsm[self._LATENT_QZV_KEY] = qzv
        # Minify test data
        if minify:
            log.info('Minifying test data.')
            self.minify_query(test_ad)
        # Get model predictions for test data
        use_full = None if not incl_unseen else True
        prediction_output = self.predict(adata=test_ad, use_full_cls_emb=use_full)
        # Select aligned output if available
        if PREDICTION_KEYS.ALIGN_SOFT_PREDICTION_KEY in prediction_output:
            # Set aligned predictions as final predictions
            soft_pred_key = PREDICTION_KEYS.ALIGN_SOFT_PREDICTION_KEY
            pred_key = PREDICTION_KEYS.ALIGN_PREDICTION_KEY
        # Fall back to standard classification
        else:
            if incl_unseen:
                log.warning(f'Unseen classes are included. Default classification output cannot zero-shot.')
            soft_pred_key = PREDICTION_KEYS.SOFT_PREDICTION_KEY
            pred_key = PREDICTION_KEYS.PREDICTION_KEY
        # Get predictions from model output
        soft_predictions = prediction_output[soft_pred_key]
        predictions = prediction_output[pred_key]
        # Save predictions to test adata
        test_ad.obs[PREDICTION_KEYS.PREDICTION_KEY] = predictions
        test_ad.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = soft_predictions
        # Evaluate test results
        if output_dir is None and getattr(self, 'model_log_dir', None) is None:
            log.warning(f'No output location provided. Please specify an output directory.')
            return
        # Create output directory, fall back to model's tensorboard directory
        if output_dir is None:
            output_dir = os.path.join(self.model_log_dir, 'test', ds_name)
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Evaluating results for test set. Test output dir: {output_dir}')
        summary, report = pf.get_classification_report(
            test_ad,
            cls_label=cls_label,
            pred_label=PREDICTION_KEYS.PREDICTION_KEY
        )
        # Save results to test adata
        test_ad.uns[PREDICTION_KEYS.SUMMARY_KEY] = summary
        test_ad.uns[PREDICTION_KEYS.REPORT_KEY] = report
        # Add split information to test data
        test_ad.obs[REGISTRY_KEYS.SPLIT_KEY] = 'test'
        # Calculate top n predictions
        top_n_predictions = pf.compute_top_n_predictions(
            test_ad,
            split_key=REGISTRY_KEYS.SPLIT_KEY,
            context_key=batch_label,
            labels_key=cls_label,
            ctrl_key=ctrl_key,
            train_perturbations=list(train_cls)
        )
        # Save top n predictions to test adata
        test_ad.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY] = top_n_predictions
        
        # Ensure batch and class keys are consistent
        test_ad.obs[self.original_batch_key] = test_ad.obs[batch_label].values
        test_ad.obs[self.original_label_key] = test_ad.obs[cls_label].values
        # Merge latent spaces and re-calculte a shared umap
        test_uns = deepcopy(test_ad.uns)
        test_ad = ad.concat((self.adata, test_ad))
        test_ad.uns = test_uns
        # Always re-calculate umap
        pl.calc_umap(test_ad, rep=self._LATENT_QZM_KEY, slot_key=self._LATENT_UMAP, force=True)
        # Determine result object
        return_obj = None
        if results_mode is not None:
            if 'return' in results_mode:
                # Return merged data adata
                return_obj = test_ad
            if 'save' in results_mode:
                # Save results adata to disk
                test_ad_o = os.path.join(output_dir, f'full.h5ad')
                test_ad.write_h5ad(test_ad_o)
        # Plot results
        if plot:
            # Create plot output directory
            plt_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plt_dir, exist_ok=True)
            # Plot top n metrics
            top_n_o = os.path.join(plt_dir, f'top_n_predictions_split.svg')
            pl.plot_top_n_performance(
                top_n_predictions=top_n_predictions,
                output_file=top_n_o,
                top_n=top_n,
                mean_split='test'
            )
            # Plot test heatmap
            hm_o = os.path.join(plt_dir, f'confusion_matrix.png')
            pl.plot_confusion(
                y_true=test_ad.obs[cls_label].values.astype(str),
                y_pred=predictions, 
                plt_file=hm_o
            )
            # Plot merged latent space for different labels
            hues = [REGISTRY_KEYS.SPLIT_KEY, self.original_batch_key, self.original_label_key]
            pl.plot_umaps(
                test_ad,
                slot=self._LATENT_UMAP,
                hue=hues,
                output_dir=plt_dir
            )
        return return_obj

    def get_training_classes(self) -> np.ndarray:
        """Get trained class label names. Class labels are sorted by observed and unobserved."""
        return self.idx_to_label[:self.n_train_labels]

    @classmethod
    def load_checkpoint(
        cls,
        model_dir: str,
        adata: AnnData | None = None,
        n: int = 0,
        checkpoint_dirname: str = 'checkpoints',
        model_name: str = 'model.pt',
        default_dirname: str = 'model'
    ):
        import os
        import glob
        # Look for checkpoints
        checkpoint_dir = os.path.join(model_dir, checkpoint_dirname)
        if os.path.exists(checkpoint_dir) and n > -1:
            checkpoint_model_paths = glob.glob(f'{checkpoint_dir}/**/{model_name}')
            if len(checkpoint_model_paths) > 0:
                checkpoint_model_p = checkpoint_model_paths[n] if n > 0 and n < len(checkpoint_model_paths) else checkpoint_model_paths[0]
                log.info(f'Loading model checkpoint {checkpoint_model_p}.')
                checkpoint_model_dir = os.path.dirname(checkpoint_model_p)
                return super().load(checkpoint_model_dir, adata=adata)
        log.info(f'Could not find model checkpoint(s). Using default "{default_dirname}" directory.')    
        model_state_dir = os.path.join(model_dir, default_dirname)
        return super().load(model_state_dir, adata=adata)
    
    def save(self, *args, **kwargs) -> None:
        """Save wrapper to handle external model embeddings"""
        # Handle external model embeddings if anndata is saved with model
        if self.cls_emb is not None and bool(kwargs.get('save_anndata')):
            # Convert embeddings to numpy arrays
            if bool(kwargs.get('keep_emb')):
                self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY] = np.array(self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY])
                self.adata.uns[REGISTRY_KEYS.CLS_SIM_KEY] = np.array(self.adata.uns[REGISTRY_KEYS.CLS_SIM_KEY])
            # Remove embeddings completely
            else:
                self.adata.uns.pop(REGISTRY_KEYS.CLS_EMB_KEY, None)
                self.adata.uns.pop(REGISTRY_KEYS.CLS_SIM_KEY, None)
        # Save model as you would normally
        super().save(*args, **kwargs)
    
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str | None = None,
        batch_key: str | None = None,
        class_emb_uns_key: str | None = 'cls_embedding',
        context_emb_uns_key: str | None = None,
        gene_emb_varm_key: str | None = None,
        size_factor_key: str | None = None,
        class_certainty_key: str | None = None,
        cast_to_csr: bool = True,
        raw_counts: bool = True,
        layer: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Setup AnnData object for training a JEDVI model.
        """
        if not isinstance(adata.X, sp.csr_matrix) and cast_to_csr:
            log.info('Converting adata.X to csr matrix to boost training efficiency.')
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
            log.info(f'Registered class embedding from adata.uns[{class_emb_uns_key}].')
            anndata_fields.append(StringUnsField(REGISTRY_KEYS.CLS_EMB_KEY, class_emb_uns_key))
        # Add context embedding matrix with shape (n_contexts, context_emb_dim)
        if context_emb_uns_key is not None and context_emb_uns_key in adata.uns:
            log.info(f'Registered context embedding from adata.uns[{context_emb_uns_key}].')
            anndata_fields.append(StringUnsField(REGISTRY_KEYS.CTX_EMB_KEY, context_emb_uns_key))
        # Add gene embedding matrix with shape (n_vars, emb_dim)
        if gene_emb_varm_key is not None and gene_emb_varm_key in adata.varm:
            log.info(f'Registered gene embedding from adata.varm[{gene_emb_varm_key}].')
            anndata_fields.append(VarmField(REGISTRY_KEYS.GENE_EMB_KEY, gene_emb_varm_key))
        # Add class score per cell, ensure its scaled [0, 1]
        if class_certainty_key is not None:
            log.info(f'Registered class certainty score from adata.obs[{class_certainty_key}].')
            scaled_class_cert_key = f'{class_certainty_key}_scaled'
            adata.obs[scaled_class_cert_key] = scale_1d_array(adata.obs[class_certainty_key].astype(float).values)
            anndata_fields.append(NumericalObsField(REGISTRY_KEYS.CLS_CERT_KEY, scaled_class_cert_key, required=False))
        # Register latent fields of adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        # Create AnnData manager
        adata_manager = EmbAnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
