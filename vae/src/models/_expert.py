import os
from src.utils.constants import REGISTRY_KEYS, EXT_CLS_EMB_INIT, PREDICTION_KEYS, MODULE_KEYS
import src.utils.io as io
import src.utils.performance as pf
from src.utils._callbacks import DelayedEarlyStopping, PeriodicTestCallback
import src.utils.plotting as pl
from src.modules._xpert import XPert
from src.modules._splitter import ContrastiveDataSplitter
from src._train.expert_plans import ContrastiveSupervisedTrainingPlan
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


class ExPert(
        RNASeqMixin, 
        VAEMixin,
        ArchesMixin, 
        UnsupervisedTrainingMixin,
        BaseMinifiedModeModelClass
    ):
    _module_cls = XPert
    _name = __qualname__
    _training_plan_cls = ContrastiveSupervisedTrainingPlan
    _LATENT_QZM_KEY = f"{__qualname__}_latent_qzm"
    _LATENT_QZV_KEY = f"{__qualname__}_latent_qzv"
    _LATENT_Z_KEY = f"{__qualname__}_latent_z"
    _LATENT_Z_SHARED_KEY = f"{__qualname__}_latent_z_shared"
    _LATENT_CTX_PROJ_KEY = f"{__qualname__}_latent_ctx_proj"
    _LATENT_CLS_PROJ_KEY = f"{__qualname__}_latent_cls_proj"
    _LATENT_Z_SHARED_UMAP = f"{__qualname__}_latent_h_umap"
    _LATENT_UMAP = f"{__qualname__}_latent_umap"


    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers: int = 2,
        n_shared: int = 128,
        dropout_rate: float = 0.2,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        use_full_cls_emb: bool = False,
        ctrl_class: str | None = None,
        **model_kwargs,
    ):
        super().__init__(adata)
        self._model_kwargs = dict(model_kwargs)
        self.use_gene_emb = self.adata_manager.registry.get('field_registries', {}).get(REGISTRY_KEYS.GENE_EMB_KEY, None) is not None
        # Set number of classes
        self.n_labels = self.summary_stats.n_labels
        # Whether to use the full class embedding
        self.use_full_cls_emb = use_full_cls_emb
        # Create placeholder for log directory
        self.model_log_dir = None
        self.ctrl_class = ctrl_class

        # Initialize indices and labels for this VAE
        self._setup()
        # Get covariates
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        # Determine number of batches/datasets and fix library method
        n_batch = self.summary_stats.n_batch
        # Check number of hidden neurons, set to input features if < 0 else use given number
        n_hidden = self.summary_stats.n_vars if n_hidden < 0 else n_hidden
        # Get class embeddings
        cls_emb, cls_sim = self.get_cls_emb()
        # Initialize model
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.n_labels,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_shared=n_shared,
            cls_emb=cls_emb,
            cls_sim=cls_sim,
            ctrl_class_idx=self.ctrl_class_idx,
            ctx_emb=self.ctx_emb,
            dropout_rate=dropout_rate,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            **self._model_kwargs,
        )
        # Fresh initialization, set params accordingly
        self.supervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.is_evaluated = False
        # Give model summary
        self.n_unseen_labels = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['n_unseen_labels'] if REGISTRY_KEYS.CLS_EMB_INIT in adata.uns else None
        self._model_summary_string = (
            f"{self.__class__} Model with the following params: \n"
            f"n_classes: {self.n_labels}, "
            f"n_unseen_classes: {self.n_unseen_labels}, \n"
            f"n_contexts: {n_batch}, "
            f"n_unseen_contexts: {self.n_unobs_ctx}, \n" if self.n_unobs_ctx > 0 else "\n"
            f"use_gene_emb: {self.use_gene_emb}"
        )
        # Include control class information
        if self.ctrl_class is not None and self.ctrl_class_idx is not None:
            self._model_summary_string += f"\nctrl_class: {self.ctrl_class}"

        
    def _setup(self):
        """Setup adata for training. Prepare embeddings."""
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
        if len(ctx_emb_registry) == 0:
            raise ValueError('ExPert requires context embeddings to be registered.')
        # Get registry keys for context embedding in adata
        ext_ctx_emb_key = ctx_emb_registry.get('original_key')
        ctx_emb_key = ctx_emb_registry['data_registry'].get('attr_key') if ext_ctx_emb_key is None else ext_ctx_emb_key
        # Get the embedding from adata and check if it's a dataframe
        if not isinstance(self.adata.uns[ctx_emb_key], pd.DataFrame):
            raise ValueError(f'Context embedding has to be a dataframe with contexs as index, got {ctx_emb.__class__}.')

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
        # Set context labels appropriately
        self.idx_to_batch = ctx_emb.index.values
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
        if len(cls_emb_registry) == 0:
            raise ValueError('ExPert requires class embeddings to be registered.')
        # Get adata.uns key for embedding, fall back to attribute key if no original key is given
        ext_cls_emb_key = cls_emb_registry.get('original_key')
        cls_emb_key = cls_emb_registry['data_registry'].get('attr_key') if ext_cls_emb_key is None else ext_cls_emb_key
        # Check if adata has already been registered with this model class
        if REGISTRY_KEYS.CLS_EMB_INIT in self.adata.uns and self._name == self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.MODEL_KEY]:
            log.info(f'Adata has already been initialized with {self.__class__}, loading model settings from adata.')
            # Set to embeddings found in adata
            self.cls_emb = io.to_tensor(self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY])
            self.cls_sim = io.to_tensor(self.adata.uns.get(REGISTRY_KEYS.CLS_SIM_KEY))
            # Init train embeddings from adata
            self.n_train_labels = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.N_TRAIN_LABELS_KEY]
            self.train_cls_emb = self.cls_emb[:self.n_train_labels,:]
            if self.cls_sim is not None:
                self.train_cls_sim = self.cls_sim[:self.n_train_labels,:self.n_train_labels]
            else:
                self.train_cls_sim = None
            # Set indices
            self.idx_to_label = np.array(self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.LABELS_KEY])
            # Set control class index
            self.ctrl_class = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.CTRL_CLASS_KEY]
            self.ctrl_class_idx = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT][EXT_CLS_EMB_INIT.CTRL_CLASS_IDX_KEY]
        else:
            # Register adata's class embedding with this model
            cls_emb: pd.DataFrame = self.adata.uns[cls_emb_key]
            if not isinstance(cls_emb, pd.DataFrame):
                raise ValueError(f'Class embedding has to be a dataframe with labels as index, got {cls_emb.__class__}.')
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
            # Set gene embedding as class parameter
            self.cls_emb = torch.tensor(cls_emb.values, dtype=torch.float32)
            self.cls_sim = None
            # Save in adata for caching
            self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY] = self.cls_emb
            # Save train embeddings and similarities as class parameters
            self.train_cls_emb = self.cls_emb[:self.n_train_labels,:]
            self.train_cls_sim = None
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

    def print_summary(self, max_depth: int = 3) -> None:
        """Print overview of module"""
        from pytorch_lightning.utilities.model_summary import ModelSummary
        summary = ModelSummary(self.module, max_depth=max_depth)
        log.info(summary)

    def draw_module(self, **kwargs) -> None:
        """Draw """
        pass

    def get_ctx_emb(self, return_dataframe: bool = False):
        """Getter function for context embedding"""
        ctx_emb = self.ctx_emb
        if ctx_emb is None:
            return None
        if return_dataframe:
            return io.tensor_to_df(ctx_emb, index=self.idx_to_batch)
        else:
            return ctx_emb

    def get_cls_emb(self, use_full_cls_emb: bool | None = None, return_dataframe: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Helper getter to return either training or full class embedding"""
        if self.cls_emb is None:
            return None, None
        # Check if we want to train on full or observed embedding only
        use_full_cls_emb = use_full_cls_emb if use_full_cls_emb is not None else self.use_full_cls_emb
        if use_full_cls_emb:
            cls_emb, cls_sim = self.cls_emb, self.cls_sim
        else:
            cls_emb, cls_sim = self.train_cls_emb, self.train_cls_sim
        # Return copy of tensor as dataframe
        if return_dataframe:
            return io.tensor_to_df(cls_emb, index=self.idx_to_label), None
        else:
            return cls_emb, cls_sim
        
    def reset_emb_to_df(self) -> None:
        """Reset model's embeddings from tensors to dataframes"""
        self.ctx_emb = self.get_ctx_emb(return_dataframe=True)
        self.cls_emb, _ = self.get_cls_emb(return_dataframe=True)
    
    @classmethod
    def from_base_model(
        cls,
        pretrained_model,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        excl_setup_keys: list[str] = ['class_emb_uns_key'],
        excl_states: list[str] | None = None,
        freeze_modules: list[str] | None = ['z_encoder', 'decoder'],
        hard_freeze: bool = False,
        check_model_kwargs: bool = True,
        **model_kwargs,
    ):
        """Initialize ExPert model with weights from pretrained :class:`~src.models.JEDVI` model.

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
        # Switch pre-training stage off
        model_params['pretrain'] = False
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
            model.freeze_modules = freeze_modules
            # Hard freeze
            if hard_freeze:
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
        zs_shared: list[Tensor] = []
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
                # Collect distribution parameters
                qz_means.append(qzm.cpu())
                qz_vars.append(qzv.cpu())
                # Get z latent space
                z: Tensor = qzm if give_mean else outputs.get(MODULE_KEYS.Z_KEY)
                # Get shared latent space
                zshared: Tensor = outputs.get(MODULE_KEYS.Z_SHARED_KEY)
                if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                    samples = qz.sample([mc_samples])
                    z = softmax(samples, dim=-1).mean(dim=0)
                # Collect tensors
                zs.append(z.cpu())
                if zshared is not None:
                    zs_shared.append(zshared.cpu())
        # Return output dictionary
        o = {
            MODULE_KEYS.Z_KEY: torch.cat(zs).numpy(),
        }
        # Add shared latent space if aligner is on
        if len(zs_shared) > 0:
            o[MODULE_KEYS.Z_SHARED_KEY] = torch.cat(zs_shared).numpy()
        # Return latent distributions if possible
        if return_dist:
            o.update({
                MODULE_KEYS.QZM_KEY: torch.cat(qz_means).numpy(), 
                MODULE_KEYS.QZV_KEY: torch.cat(qz_vars).numpy()
            })
        return o
        
    def validate_ctx_emb(self, ctx_emb: torch.Tensor):
        # Extract labels if it's a dataframe
        if isinstance(ctx_emb, pd.DataFrame):
            labels = ctx_emb.index.values
        else:
            labels = None
        # Convert to tensor
        ctx_emb = io.to_tensor(ctx_emb)
        assert self.module.n_ctx_dim != ctx_emb.shape[-1], f'Dimension mismatch, ctx_emb has to have {self.module.n_ctx_dim} dimensions.'
        return labels, ctx_emb

    def validate_cls_emb(self, cls_emb: torch.Tensor):
        # Extract labels if it's a dataframe
        if isinstance(cls_emb, pd.DataFrame):
            labels = cls_emb.index.values
        else:
            labels = None
        # Convert to tensor
        cls_emb = io.to_tensor(cls_emb)
        assert self.module.n_cls_dim != cls_emb.shape[-1], f'Dimension mismatch, cls_emb has to have {self.module.n_cls_dim} dimensions.'
        return labels, cls_emb

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        use_posterior_mean: bool | None = None,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        return_latents: bool = True,
        inference: bool = False,
        verbose: bool = True,
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
        from tqdm import tqdm
        # Check if model has been trained
        self._check_if_trained(warn=False)
        # Log usage of gene embeddings
        if self.gene_emb is not None:
            log.info(f'Using model gene embeddings ({self.gene_emb.shape})')
        # validate adata or get it from model
        adata = self._validate_anndata(adata)
        # Take all cells by default
        if indices is None:
            indices = np.arange(adata.n_obs)
        # Create dataloader
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size
        )
        
        # Use new context embedding at inference
        if ctx_emb is not None:
            log.info(f'Using new context embedding: {ctx_emb.shape}')
            ctx_labels, ctx_emb = self.validate_ctx_emb(ctx_emb).to(self.device)
        # Use model's context labels and embedding
        else:
            ctx_labels = self.idx_to_batch
            ctx_emb = self.module.ctx_emb.weight
        # Use new class embedding at inference
        if cls_emb is not None:
            log.info(f'Using new class embedding: {cls_emb.shape}')
            cls_labels, cls_emb = self.validate_cls_emb(cls_emb).to(self.device)
        # Use model's class labels and embedding
        else:
            cls_labels = self.idx_to_label
            cls_emb = self.module.cls_emb.weight

        # Add control embedding
        if self.module.ctrl_class_idx is not None:
            log.info('Adding control embedding')
            if self.module.ctrl_class_idx >= cls_emb.size(0):
                pad = torch.zeros((1, cls_emb.size(-1)), device=cls_emb.device)
                cls_emb = torch.cat((cls_emb, pad), dim=0)
            else:
                # Reset index to 0s
                cls_emb[self.module.ctrl_class_idx] = torch.tensor(0.0)
   
        # Use aligner classify mode
        if inference:
            log.info(f'Using inference mode (joint prediction).')
            # Register embedding interactions
            self.module.aligner.precompute_joint_bank(ctx_emb, cls_emb)
        
        log.info(f'Classifying.')
        # Collect batch logits
        ctx_logits = []
        cls_logits = []
        z_cls_logits = []
        # Collect batch z shared
        zs = []
        z_shared = []
        ctx2z = []
        cls2z = []
        # Collect batch labels
        data_ctx_labels = []
        data_cls_labels = []
        # Add progress bar to classification if toggled
        if verbose:
            dl_it = tqdm(enumerate(scdl))
        else:
            dl_it = enumerate(scdl)
        # Run through each batch of data
        with torch.no_grad():
            for _, tensors in dl_it:
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
                        MODULE_KEYS.QZM_KEY: tensors[self._LATENT_QZM_KEY],
                        MODULE_KEYS.QZV_KEY: tensors[self._LATENT_QZV_KEY], 
                    }
                else:
                    inference_outputs = None
                # Run classification process, i.e. full / cached / partial inference
                cls_output = self.module.classify(
                    x,
                    batch_index=batch,
                    label=l,
                    g=self.gene_emb,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    use_posterior_mean=use_posterior_mean,
                    inference_outputs=inference_outputs,
                    inference=inference,
                    ctx_emb=ctx_emb,
                    cls_emb=cls_emb,
                    return_logits=True
                )
                # Get regular classifier logits
                _z_cls_logits = cls_output['logits'].cpu()
                # Unpack context and perturbation classification output
                _ctx_logits = cls_output[MODULE_KEYS.CTX_LOGITS_KEY].cpu()
                _cls_logits = cls_output[MODULE_KEYS.CLS_LOGITS_KEY].cpu()
                # Add batch predictions to overall predictions
                z_cls_logits.append(_z_cls_logits)
                ctx_logits.append(_ctx_logits)
                cls_logits.append(_cls_logits)
                # Unpack shared latent spaces
                _z = cls_output[MODULE_KEYS.Z_KEY].cpu()
                _z_shared = cls_output[MODULE_KEYS.Z_SHARED_KEY].cpu()
                _ctx2z = cls_output[MODULE_KEYS.CTX_PROJ_KEY].cpu()
                _cls2z = cls_output[MODULE_KEYS.CLS_PROJ_KEY].cpu()
                zs.append(_z)
                z_shared.append(_z_shared)
                ctx2z.append(_ctx2z)
                cls2z.append(_cls2z)
                # Collect labels
                data_ctx_labels.append(batch.cpu())
                data_cls_labels.append(l.cpu())
                # ---- CLEANUP ----
                del cls_output, x, l, batch
                torch.cuda.empty_cache()
        
        log.info(f'Aggregating predictions')
        # Concatenate batch results
        z_cls_logits = torch.cat(z_cls_logits).numpy()
        ctx_logits = torch.cat(ctx_logits).numpy()
        cls_logits = torch.cat(cls_logits).numpy()
        zs = torch.cat(zs).numpy()
        z_shared = torch.cat(z_shared).numpy()
        ctx2z = torch.stack(ctx2z, dim=0).mean(0).numpy()
        cls2z = torch.stack(cls2z, dim=0).mean(0).numpy()
        data_ctx_labels = torch.cat(data_ctx_labels).numpy()
        data_cls_labels = torch.cat(data_cls_labels).numpy()

        # Set nans to 0s
        ctx_logits = np.nan_to_num(ctx_logits, -np.inf)
        cls_logits = np.nan_to_num(cls_logits, -np.inf)

        # Add classifier predictions
        z_cls_soft_predictions = pd.DataFrame(
            z_cls_logits,
            columns=cls_labels[:z_cls_logits.shape[-1]],
            index=adata.obs_names[indices],
        )
        z_cls_predictions = z_cls_soft_predictions.columns[np.argmax(z_cls_soft_predictions.values, axis=-1)]

        # Add context predictions
        ctx_soft_predictions = pd.DataFrame(
            ctx_logits,
            columns=ctx_labels[:ctx_logits.shape[-1]],
            index=adata.obs_names[indices],
        )
        ctx_predictions = ctx_soft_predictions.columns[np.argmax(ctx_soft_predictions.values, axis=-1)]
        # Add class predictions
        cls_soft_predictions = pd.DataFrame(
            cls_logits,
            columns=cls_labels[:cls_logits.shape[-1]],
            index=adata.obs_names[indices],
        )
        cls_predictions = cls_soft_predictions.columns[np.argmax(cls_soft_predictions.values, axis=-1)]
        
        # Return all model predictions
        o = {
            PREDICTION_KEYS.Z_PREDICTION_KEY: z_cls_predictions,
            PREDICTION_KEYS.Z_SOFT_PREDICTION_KEY: z_cls_soft_predictions,
            PREDICTION_KEYS.CTX_PREDICTION_KEY: ctx_predictions,
            PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY: ctx_soft_predictions,
            PREDICTION_KEYS.PREDICTION_KEY: cls_predictions,
            PREDICTION_KEYS.SOFT_PREDICTION_KEY: cls_soft_predictions,
            REGISTRY_KEYS.BATCH_KEY: data_ctx_labels,
            REGISTRY_KEYS.LABELS_KEY: data_cls_labels,
            'indices': indices,
        }
        # Add latent spaces to prediction output
        if return_latents:
            o.update({
                MODULE_KEYS.Z_KEY: zs,
                MODULE_KEYS.Z_SHARED_KEY: z_shared,
                MODULE_KEYS.CTX_PROJ_KEY: ctx2z,
                MODULE_KEYS.CLS_PROJ_KEY: cls2z,
            })
        return o
        
    def get_training_plan(self, **plan_kwargs):
        # If fine-tune stage, set lower lr for encoder and decoder
        soft_freeze_lr = plan_kwargs.pop('soft_freeze_lr', None)
        freeze_modules = getattr(self, 'freeze_modules', None)
            
        # Create training plan
        return self._training_plan_cls(
            module=self.module, 
            n_classes=self.n_labels, 
            gene_emb=self.gene_emb,
            freeze_modules=freeze_modules,
            soft_freeze_lr=soft_freeze_lr,
            **plan_kwargs
        )
    
    @property
    def is_minified(self) -> bool:
        return _get_adata_minify_type(self.adata) is not None
    
    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        from src.tune._statics import CONF_KEYS, NESTED_CONF_KEYS
        return {
            CONF_KEYS.DATA: {},
            CONF_KEYS.MODEL: {
                NESTED_CONF_KEYS.ENCODER_KEY,
                NESTED_CONF_KEYS.DECODER_KEY,
                NESTED_CONF_KEYS.ALIGN_KEY,
            },
            CONF_KEYS.TRAIN: {
                NESTED_CONF_KEYS.PLAN_KEY: {
                    NESTED_CONF_KEYS.SCHEDULES_KEY
                }
            },
        }
        
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

        # Look for test contexts
        if 'test_context_labels' in data_params:
            log.info(f'Using {data_params.get("test_context_labels")} as test context(s).')
        
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
        # Add test callback
        check_test_every_n_epoch = train_params.pop('check_test_every_n_epoch')
        if check_test_every_n_epoch is not None:
            test_callback = PeriodicTestCallback(every_n_epochs=check_test_every_n_epoch)
            callbacks.append(test_callback)
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
        # TODO: Test call
        # test_out = runner.trainer.test(self, datamodule=data_splitter)
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
        reports = pd.concat(reports, axis=0).reset_index(names=['cls_label'])
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
        latent_out = self.get_latent_representation(return_dist=True)
        self.adata.obsm[self._LATENT_QZM_KEY] = latent_out[MODULE_KEYS.QZM_KEY]
        self.adata.obsm[self._LATENT_QZV_KEY] = latent_out[MODULE_KEYS.QZV_KEY]

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
        self.adata.obs[REGISTRY_KEYS.SPLIT_KEY] = pd.Categorical(split_labels)
    
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
        po = self.predict(return_latents=True)
        # Save context predictions
        self.adata.obs[PREDICTION_KEYS.CTX_PREDICTION_KEY] = po[PREDICTION_KEYS.CTX_PREDICTION_KEY]
        self.adata.obsm[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY] = po[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY]
        # Save regular classification predictions to model adata
        if self.module.align_ext_emb_loss_strategies is None:
            # Use classifier predictions
            predictions = pd[PREDICTION_KEYS.Z_PREDICTION_KEY]
            soft_predictions = po[PREDICTION_KEYS.Z_SOFT_PREDICTION_KEY]
        else:
            # Use aligner predictions
            predictions = po[PREDICTION_KEYS.PREDICTION_KEY]
            soft_predictions = po[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
        self.adata.obs[PREDICTION_KEYS.PREDICTION_KEY] = predictions
        self.adata.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = soft_predictions
        # Register aligned latent spaces with model's adata
        self.adata.obsm[self._LATENT_Z_KEY] = po[MODULE_KEYS.Z_KEY]
        self.adata.obsm[self._LATENT_Z_SHARED_KEY] = po[MODULE_KEYS.Z_SHARED_KEY]
        self.adata.uns[self._LATENT_CTX_PROJ_KEY] = po[MODULE_KEYS.CTX_PROJ_KEY]
        self.adata.uns[self._LATENT_CLS_PROJ_KEY] = po[MODULE_KEYS.CLS_PROJ_KEY]
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
            z_shared: bool = True,
        ) -> dict[pd.DataFrame] | None:
        """Run model evaluation for all available data splits registered with the model."""
        if self.is_evaluated and not force:
            log.info('This model has already been evaluated, pass force=True to re-evaluate.')
            return
        # Set z shared to false if model is in pretrain mode
        if self.module.pretrain:
            z_shared = False
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
            self._plot_evalutation(output_dir=base_output_dir, z_shared=z_shared)
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
        # Plot latent projections

    def _plot_eval_splits(self, plt_dir: str) -> None:
        """Generate all split-specific plots."""
        for split in self._get_available_splits():
            self._plot_eval_split(split, plt_dir=plt_dir)
    
    def _plot_evalutation(self, output_dir: str | None, z_shared: bool = True, metric: str = 'f1-score') -> None:
        """Save plots associated to model evaluation."""
        if output_dir is None:
            log.warning('Evalutaion output directory is not available. Skipping plots.')
            return
        # Set plotting directory and create if needed
        plt_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plt_dir, exist_ok=True)
        # Save latent space to model adata if not already calculated
        self._register_latent_variables(force=False)
        # Plot performance - support correlations
        support_corr_o = os.path.join(plt_dir, f'support_correlations.png')
        pl.plot_performance_support_corr(self.adata.uns[PREDICTION_KEYS.REPORT_KEY], o=support_corr_o, hue=REGISTRY_KEYS.SPLIT_KEY)
        # Plot performance metric over top N predictions in all splits
        top_n_o = os.path.join(plt_dir, f'top_n_{metric}.png')
        pl.plot_top_n_performance(
            self.adata.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY],
            output_file=top_n_o,
            metric=metric,
            mean_split='test'
        )
        z_key = self._LATENT_Z_SHARED_KEY if z_shared else self._LATENT_Z_KEY
        # Calculate UMAP for latent space based on latent mean
        pl.calc_umap(self.adata, rep=z_key, slot_key=self._LATENT_UMAP)
        # Plot full UMAP colored by data split
        umap_split_o = os.path.join(plt_dir, f'full_umap_{REGISTRY_KEYS.SPLIT_KEY}.png')
        pl.plot_umap(self.adata, slot=self._LATENT_UMAP, hue=REGISTRY_KEYS.SPLIT_KEY, output_file=umap_split_o)
        # Plot full UMAP colored by batch label
        umap_batch_o = os.path.join(plt_dir, f'full_umap_{self.original_batch_key}.png')
        pl.plot_umap(self.adata, slot=self._LATENT_UMAP, hue=self.original_batch_key, output_file=umap_batch_o)
        # Plot full UMAP colored by batch label
        umap_label_o = os.path.join(plt_dir, f'full_umap_{self.original_label_key}.png')
        pl.plot_umap(self.adata, slot=self._LATENT_UMAP, hue=self.original_label_key, output_file=umap_label_o)
        # TODO: Plot shared z umap with context and class embedding markers on it
        # Combine z_shared with 
        # Plot individual splits
        self._plot_eval_splits(plt_dir)
        return
    
    def align_model_features(self, adata: AnnData, inplace: bool = True) -> AnnData:
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
        _adata = adata[:,self.adata.var_names]
        # Inplace subset adata vars
        if inplace:
            adata = _adata.copy()
            return
        # Return updated adata
        else:
            return _adata
    
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
        batch_label: str = 'context',
        output_dir: str | None = None,
        incl_unseen: bool = False,
        plot: bool = True,
        results_mode: Literal['return', 'save'] | None | list[str] = 'save',
        top_n: int = 10,
        ctrl_key: str | None = 'control',
        minify: bool = False,
        use_pathways: bool = False,
        **kwargs
    ) -> None:
        """Function to evaluate model on unseen data."""
        # Refactor results mode to always be a list of options or None
        results_mode: list[str] | None = [results_mode] if results_mode is not None and not isinstance(results_mode, list) else None
        # TODO: all this only works for labelled data, implement option to handle unlabelled data
        # TODO: actually pretty easy, just set prediction column to unknown for all cells duh
        # TODO: make this more modular
        # Read test adata and filter columns
        log.info(f'Testing model with: {test_adata_p}')
        test_ad = io.read_adata(test_adata_p, cls_label=cls_label, **kwargs)
        # Filter test adata for trained classes
        train_cls = set(self.get_training_classes())
        test_cls = set(test_ad.obs[cls_label].unique())
        available_cls = set(self.idx_to_label)
        log.info(f'Available classes for prediction: {len(available_cls)}, #observed: {len(train_cls)}')
        # Look for shared classes between testing set and trained data
        shared_cls = test_cls.intersection(train_cls)
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
        # Save original labels
        test_labels = test_ad.obs[cls_label].values
        # Register new testing data with model
        self.setup_anndata(test_ad, labels_key=cls_label, batch_key=batch_label)
        # Make sure test adata's features are consistent with model
        test_ad = self.align_model_features(test_ad, inplace=False).copy()
        # Get latent representation of model
        latent_out = self.get_latent_representation(adata=test_ad, return_dist=True)
        test_ad.obsm[self._LATENT_QZM_KEY] = latent_out[MODULE_KEYS.QZM_KEY]
        test_ad.obsm[self._LATENT_QZV_KEY] = latent_out[MODULE_KEYS.QZV_KEY]
        # Minify test data
        if minify:
            log.info('Minifying test data.')
            self.minify_query(test_ad)
        # Get model predictions for test data
        use_full = None if not incl_unseen else True
        prediction_output = self.predict(
            adata=test_ad, 
            use_full_cls_emb=use_full, 
            return_latents=True,
        )
        # Save predictions to test adata
        predictions = prediction_output[PREDICTION_KEYS.PREDICTION_KEY]
        test_ad.obs[PREDICTION_KEYS.PREDICTION_KEY] = predictions
        test_ad.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = prediction_output[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
        # Save context predictions to test set
        test_ad.obs[PREDICTION_KEYS.CTX_PREDICTION_KEY] = prediction_output[PREDICTION_KEYS.CTX_PREDICTION_KEY]
        test_ad.obsm[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY] = prediction_output[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY]
        # Save shared latent spaces to test set
        test_ad.obsm[self._LATENT_Z_SHARED_KEY] = prediction_output[MODULE_KEYS.Z_SHARED_KEY]
        test_ad.uns[self._LATENT_CTX_PROJ_KEY] = prediction_output[MODULE_KEYS.CTX_PROJ_KEY]
        test_ad.uns[self._LATENT_CLS_PROJ_KEY] = prediction_output[MODULE_KEYS.CLS_PROJ_KEY]
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
            train_perturbations=list(train_cls),
            use_pathways=use_pathways
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
        pl.calc_umap(test_ad, rep=self._LATENT_Z_SHARED_KEY, slot_key=self._LATENT_UMAP, force=True)
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
            # Add another split to data if unseen classes are included
            if incl_unseen:
                # Copy values of split where actual values are
                ext_cls_label = 'ext_cls_label'
                no_random_mask = top_n_predictions.split!='random'
                zero_shot_mask = ~top_n_predictions['is_training_perturbation']
                top_n_predictions[ext_cls_label] = top_n_predictions.split.values
                # Set all mark zero-shot labels as such if they are not random
                ext_mask = no_random_mask & zero_shot_mask
                top_n_predictions.loc[ext_mask,ext_cls_label] = top_n_predictions.loc[ext_mask,ext_cls_label].astype(str) + '-zero-shot'
                # Set label as split hue
                hue = ext_cls_label
            else:
                hue = REGISTRY_KEYS.SPLIT_KEY
            # Plot top N performance
            pl.plot_top_n_performance(
                top_n_predictions=top_n_predictions,
                output_file=top_n_o,
                hue=hue,
                top_n=top_n,
                mean_split='test'
            )
            # Plot test only heatmap
            hm_o = os.path.join(plt_dir, f'confusion_matrix.png')
            pl.plot_confusion(
                y_true=test_labels,
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
        batch_key: str | None = None,
        labels_key: str | None = None,
        context_emb_uns_key: str | None = 'ctx_embedding',
        class_emb_uns_key: str | None = 'cls_embedding',
        gene_emb_varm_key: str | None = None,
        size_factor_key: str | None = None,
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
        # Register latent fields of adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        # Create AnnData manager
        adata_manager = EmbAnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
