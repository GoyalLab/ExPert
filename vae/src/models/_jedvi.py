import os
from src.utils.constants import REGISTRY_KEYS
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
import logging
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight


from collections.abc import Sequence
from typing import Literal, Any, Iterator

from anndata import AnnData

# scvi imports
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner, SaveCheckpoint
from scvi.data._utils import _get_adata_minify_type, _check_if_view
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

logger = logging.getLogger(__name__)


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
    _LATENT_Z2C_KEY = f"{__qualname__}_latent_z2c"
    _LATENT_C2Z_KEY = f"{__qualname__}_latent_c2z"


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
        # Whether to use the full class embedding
        self.use_full_cls_emb = use_full_cls_emb

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
            dropout_rate_encoder=dropout_rate,
            dropout_rate_decoder=dropout_rate_decoder,
            class_embed_dim=self.n_dims_emb,
            use_embedding_classifier=self.use_embedding_classifier,
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
            f"n_unseen_classes: {self.n_unseen_labels}, "
            f"use_gene_emb: {self.use_gene_emb}"
        )
        if self.ctrl_class is not None:
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
            logging.info(f'Initialized gene embedding with {emb_dim} dimensions')
            # Convert to dense if sparse
            if sp.issparse(gene_emb):
                gene_emb = gene_emb.todense()
            # Convert to tensor
            self.gene_emb = torch.Tensor(gene_emb).T

        # Assign batch labels
        batch_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        self.original_batch_key = batch_state_registry.original_key
        # Assign class embedding
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self._label_mapping = labels_state_registry.categorical_mapping
        self.n_unseen_labels = None
        self._code_to_label = dict(enumerate(self._label_mapping))
        # Check if adata has a class embedding registered
        cls_emb_registry = self.adata_manager.registry['field_registries'].get(REGISTRY_KEYS.CLS_EMB_KEY, {})
        if len(cls_emb_registry) > 0:
            # Get adata.uns key for embedding, fall back to attribute key if no original key is given
            ext_cls_emb_key = cls_emb_registry.get('original_key')
            cls_emb_key = cls_emb_registry['data_registry'].get('attr_key') if ext_cls_emb_key is None else ext_cls_emb_key
            # Check if adata has already been registered with this model class
            if REGISTRY_KEYS.CLS_EMB_INIT in self.adata.uns and self._name == self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['model']:
                logging.info(f'Adata has already been initialized with {self.__class__}, loading model settings from adata.')
                # Set to embeddings found in adata
                self.cls_emb = io.to_tensor(self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY])
                self.cls_sim = io.to_tensor(self.adata.uns[REGISTRY_KEYS.CLS_SIM_KEY])
                # Init train embeddings from adata
                n_train = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['n_train_labels']
                self.train_cls_emb = self.cls_emb[:n_train,:]
                self.train_cls_sim = self.cls_sim[:n_train,:n_train]
                # Set indices
                self.idx_to_label = np.array(self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['labels'])
                # Set control class index
                self.ctrl_class = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['ctrl_class']
                self.ctrl_class_idx = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['ctrl_class_idx']
            else:
                # Register adata's class embedding with this model
                cls_emb: pd.DataFrame = self.adata.uns[cls_emb_key]
                if not isinstance(cls_emb, pd.DataFrame):
                    logging.warning(f'Class embedding has to be a dataframe with labels as index, got {cls_emb.__class__}. Falling back to internal embedding.')
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
                        logging.warning(f'Specified control label {self.ctrl_class} is not in adata class labels, ignoring parameter.')
                    if self.ctrl_class is not None and ctrl_exists_in_data:
                        # Find control index
                        self.ctrl_class_idx = ctrl_class_idx_matches[0]
                        # Create empty embedding for control
                        if self.ctrl_class not in _shared_labels:
                            logging.info(f'Adding empty control class embedding, will be learned by model.')
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
                            logging.info(f'Overwriting existing control class embedding, will be learned by model.')
                        # Reset label overlap
                        _label_overlap = _label_series.isin(_shared_labels)
                    else:
                        # Set no label as control class
                        self.ctrl_class_idx = None
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
                    self.idx_to_label = cls_emb.index.values
                    # Save registration with this model
                    self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT] = {
                        'model': self._name,
                        'labels': self.idx_to_label,
                        'n_train_labels': self.n_train_labels, 
                        'n_unseen_labels': self.n_unseen_labels,
                        'ctrl_class': self.ctrl_class,
                        'ctrl_class_idx': self.ctrl_class_idx,
                    }
            # Set embedding dimension
            self.n_dims_emb = self.adata.uns[REGISTRY_KEYS.CLS_EMB_KEY].shape[1]
            # Use class embedding classifier
            self.use_embedding_classifier = True
        else:
            logging.info(f'No class embedding found in adata, falling back to internal embeddings with dimension 128. You can change this by specifying `n_dims_emb`.')
            self.n_dims_emb = self._model_kwargs.get('n_dims_emb', 128)
            self.ctrl_class_idx = None
            # Use base classifier
            self.use_embedding_classifier = False

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
        excl_states: list[str] = ['learned_class_embeds'],
        freeze_pretrained_base: bool = True,
        **model_kwargs,
    ):
        """Initialize jedVI model with weights from pretrained :class:`~scvi.model.JEDVI` model.

        Parameters
        ----------
        TODO: add parameter description
        """
        from copy import deepcopy
        from scvi.data._constants import (
            _SETUP_ARGS_KEY,
        )
        from scvi.data._utils import _is_minified

        pretrained_model._check_if_trained(message="Passed in scvi model hasn't been trained yet.")

        model_kwargs = dict(model_kwargs)
        init_params = pretrained_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (_, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in model_kwargs.keys():
                del model_kwargs[k]
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
            raise ValueError(
                "A `labels_key` is necessary as the scVI model was initialized without one."
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
        model = cls(adata, **non_kwargs, **kwargs, **model_kwargs)
        # Load pre-trained model weights
        pretrained_state_dict = pretrained_model.module.state_dict()
        # Remove weights that should be excluded
        pretrained_states = pd.Series(pretrained_state_dict.keys(), dtype=str)
        pretrained_state_mask = ~pretrained_states.isin(excl_states)
        pretrained_state_dict = {k: v for k in pretrained_state_dict.items() if k in pretrained_states[pretrained_state_mask]}
        # Load pre-trained weights
        model.module.load_state_dict(pretrained_state_dict, strict=False)
        # Label model as pre-trained
        model.was_pretrained = True
        # Freeze base vae module after pre-training
        if freeze_pretrained_base:
            logging.info('Freezing en- and decoder weights for fine-tuning.')
            model.module.freeze_vae_base()
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
        # Check if model has been trained
        self._check_if_trained(warn=False)
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
        if use_full_cls_emb is None or not use_full_cls_emb:
            cls_emb, _ = self.module.class_embedding(device=self.device)
        # Fall back to full model embeddings
        else:
            cls_emb = self.cls_emb

        if cls_emb is not None:
            logging.info(f'Using class embedding: {cls_emb.shape}')
        
        y_pred = []
        y_cz = []
        y_ez = []
        for _, tensors in enumerate(scdl):
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
            # Run classification process
            cls_output = self.module.classify(
                x,
                batch_index=batch,
                g=self.gene_emb,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
                class_embeds=cls_emb
            )
            # Handle output based on classifier used
            if self.module.use_embedding_classifier:
                # Unpack embedding projections
                pred, cz, ez = cls_output
                y_cz.append(cz.detach().cpu())
                y_ez.append(ez.detach().cpu())
            else:
                # Set embeddings to z, TODO: set to None or empty tensors?
                pred = cls_output
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())
        # Concatenate batch results
        y_pred = torch.cat(y_pred).numpy()
        if self.module.use_embedding_classifier:
            y_cz = torch.cat(y_cz).numpy()
            # Calculate mean embedding projections
            y_ez = torch.stack(y_ez, dim=0).mean(dim=0).numpy()
        else:
            y_cz, y_ez = None, None
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
        # Return latent projections if they are available
        if return_latent and y_cz is not None and y_ez is not None:
            return pred, y_cz, y_ez
        # Return predictions only
        else:
            return pred
        
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
        data_params: dict[str, Any]={}, 
        model_params: dict[str, Any]={}, 
        train_params: dict[str, Any]={},
        minify_adata: bool = True,
    ):
        """Train the model.

        Parameters
        ----------
        TODO: fill in parameteres
        data_params: dict
            **kwargs for src.modules._splitter.ContrastiveDataSplitter
        """
        # Get max epochs specified in params
        epochs = train_params.get('max_epochs')
        # Determine number of epochs needed for complete training
        max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
        # Train for suggested number of epochs if no specification is give
        if epochs is None:
            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))
            epochs = max_epochs
        logging.info(f'Epochs suggested: {max_epochs}, training for {epochs} epochs.')
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
        # Create training plan
        training_plan = self.get_training_plan(**plan_kwargs)
        # Check if tensorboard logger is given
        logger = train_params.get('logger')
        callbacks = []
        # Manage early stopping callback
        if train_params.get('early_stopping', False) and train_params.get('early_stopping_start_epoch') is not None:
            # Disable scvi internal early stopping
            train_params['early_stopping'] = False
            # Create custom delayed early stopping using the start epoch parameter
            early_stopping_start_epoch = train_params.pop('early_stopping_start_epoch')
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
        # Save most recent log dir
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
            logging.info(f'Saving model checkpoints to: {checkpoint_dir}')
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
        if not self.was_pretrained:
            self._model_summary_string += f", use_full_cls_emb: {self.use_full_cls_emb}"
        self.__repr__()
        # Save runner in model
        self.last_runner = runner
        # Train model
        runner()
        # Add latent representation to model
        qzm, qzv = self.get_latent_representation(return_dist=True)
        self.adata.obsm[REGISTRY_KEYS.LATENT_QZM_KEY] = qzm
        self.adata.obsm[REGISTRY_KEYS.LATENT_QZV_KEY] = qzv
        # Minify the model if enabled and adata is not already minified
        if minify_adata and not self.is_minified:
            logging.info(f'Minifying adata with latent distribution')
            self.minify_adata(use_latent_qzm_key=REGISTRY_KEYS.LATENT_QZM_KEY, use_latent_qzv_key=REGISTRY_KEYS.LATENT_QZV_KEY)
        
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
        if REGISTRY_KEYS.PREDICTION_KEY not in self.adata.obs:
            raise ValueError('Run self.evaluate() before calculating statistics.')
        # Get mode adata
        mode_adata = self._get_split_adata(split, ignore_ctrl=ignore_ctrl)
        # Generate a classification report for the relevant mode
        report = classification_report(
            mode_adata.obs[self.original_label_key],
            mode_adata.obs[REGISTRY_KEYS.PREDICTION_KEY],
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
        self.adata.uns[REGISTRY_KEYS.SUMMARY_KEY] = summaries
        self.adata.uns[REGISTRY_KEYS.REPORT_KEY] = reports

    def _register_top_n_predictions(self) -> None:
        """Calculate performance metrics for each data split and optionally also for each context"""
        top_n_predictions = pf.compute_top_n_predictions(
            adata=self.adata,
            split_key=REGISTRY_KEYS.SPLIT_KEY,
            context_key=self.original_batch_key,
            labels_key=self.original_label_key,
            ctrl_key=self.ctrl_class,
            predictions_key=REGISTRY_KEYS.SOFT_PREDICTION_KEY,
        )
        # Add data to model's adata
        self.adata.uns[REGISTRY_KEYS.TOP_N_PREDICTION_KEY] = top_n_predictions

    def _get_top_n_predictions(self) -> pd.DataFrame:
        if REGISTRY_KEYS.TOP_N_PREDICTION_KEY not in self.adata.obsm:
            raise KeyError('No model evaluation top prediction metrics available. Run model.evaluate()')
        return self.adata.obsm[REGISTRY_KEYS.TOP_N_PREDICTION_KEY]

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
        if REGISTRY_KEYS.PREDICTION_KEY in self.adata.obs and not force:
            return
        # Add split information to self.adata
        self._register_split_labels(force=force)
        # Run classifier forward pass, return soft predictions and latent spaces
        soft_pred, z2c, c2z = self.predict(soft=True, return_latent=True)
        # Save predictions to model adata
        self.adata.obs[REGISTRY_KEYS.PREDICTION_KEY] = soft_pred.columns[np.argmax(soft_pred, axis=-1)]     # (cells,)
        self.adata.obsm[REGISTRY_KEYS.SOFT_PREDICTION_KEY] = soft_pred          # (cells, classes)
        self.adata.obsm[self._LATENT_Z2C_KEY] = z2c     # z to class space projection latent (cells, shared dim)
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
            results_mode: Literal['return', 'save'] | None = 'save',
            force: bool = True,
            ignore_ctrl: bool = True,
        ) -> dict[pd.DataFrame] | None:
        """Run model evaluation for all available data splits registered with the model."""
        if self.is_evaluated and not force:
            logging.info('This model has already been evaluated, pass force=True to re-evaluate.')
            return
        # Run full prediction for all registered data
        logging.info('Running model predictions on all data splits.')
        self._register_model_predictions(force=force)
        # Run classification report for all data splits
        logging.info('Generating reports.')
        self._register_classification_report(force=force, ignore_ctrl=ignore_ctrl)
        # Save model to tensorboard logger directory if registered
        if self.model_log_dir is not None:
            base_output_dir = self.model_log_dir
            model_output_dir = os.path.join(self.model_log_dir, 'model')
        # Try to fall back to output_dir parameter and skip if not specified
        else:
            if output_dir is not None:
                base_output_dir = output_dir
                model_output_dir = os.path.join(output_dir, 'model')
            else:
                logging.warning('Could not find model tensorboard log directory or a specified "output_dir", skipping model saving.')
                base_output_dir = None
                model_output_dir = None
        # Plot evalutation results if specified
        logging.info('Plotting evaluation results.')
        if plot:
            self._plot_evalutation(output_dir=base_output_dir)
        # Save model only if we have an output directory
        if model_output_dir is not None:
            # Always save anndata if its minified else fall back to parameter
            save_anndata = True if self.is_minified else save_anndata
            save_ad_txt = 'with' if save_anndata else 'without'
            adata_txt = 'adata (minified)' if self.is_minified else 'adata'
            logging.info(f'Saving model {save_ad_txt} {adata_txt} to: {model_output_dir}')
            self.save(dir_path=model_output_dir, save_anndata=save_anndata, overwrite=True)
        # Create evalutaion return object
        return_obj = None
        if results_mode is not None:
            # Collect evaluation result metrics
            eval_report = self.get_eval_report()
            eval_summary = self.get_eval_summary()
            # Return evaluation results as dictionary
            if results_mode == 'return':
                return_obj = {
                    REGISTRY_KEYS.REPORT_KEY: eval_report,
                    REGISTRY_KEYS.SUMMARY_KEY: eval_summary,
                }
            # Save reports as seperate files in output directory
            elif results_mode == 'save':
                logging.info(f'Saving evaluation metrics to: {base_output_dir}')
                sum_o = os.path.join(base_output_dir, f'{self._name}_eval_summary.csv')
                rep_o = os.path.join(base_output_dir, f'{self._name}_eval_report.csv')
                eval_summary.to_csv(sum_o)
                eval_report.to_csv(rep_o)
        logging.info(f'Evaluation done.')
        # Return 
        return return_obj
    
    def get_eval_summary(self) -> pd.DataFrame | None:
        if REGISTRY_KEYS.SUMMARY_KEY not in self.adata.uns:
            logging.warning('No model evaluation summary available.')
            return None
        return self.adata.uns[REGISTRY_KEYS.SUMMARY_KEY]
    
    def get_eval_report(self) -> pd.DataFrame | None:
        if REGISTRY_KEYS.REPORT_KEY not in self.adata.uns:
            logging.warning('No model evaluation report available.')
            return None
        return self.adata.uns[REGISTRY_KEYS.REPORT_KEY]
    
    def _plot_eval_split(self, split: str, plt_dir: str) -> None:
        """Plot confusion matrix per split."""
        # Get split data without control cells
        adata = self._get_split_adata(split, ignore_ctrl=True)
        # Get actual and predicted class labels
        y = adata.obs[self.original_label_key].values.astype(str)
        y_hat = adata.obs[REGISTRY_KEYS.PREDICTION_KEY].values.astype(str)
        # Create output file path
        o = os.path.join(plt_dir, f'cm_{split}.png')
        pl.plot_confusion(y, y_hat, plt_file=o)

    def _plot_eval_splits(self, plt_dir: str) -> None:
        """Generate all split-specific plots."""
        for split in self._get_available_splits():
            self._plot_eval_split(split, plt_dir=plt_dir)
    
    def _plot_evalutation(self, output_dir: str | None, metric: str = 'f1-score') -> None:
        """Save plots associated to model evaluation."""
        if output_dir is None:
            logging.warning('Evalutaion output directory is not available. Skipping plots.')
            return
        # Set plotting directory and create if needed
        plt_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plt_dir, exist_ok=True)
        # Calculate UMAP for latent space based on latent mean
        umap_slot_key = f'{REGISTRY_KEYS.LATENT_QZM_KEY}_umap'
        pl.calc_umap(self.adata, rep=REGISTRY_KEYS.LATENT_QZM_KEY, slot_key=umap_slot_key)
        # Plot full UMAP colored by data split
        umap_split_o = os.path.join(plt_dir, f'full_umap_{REGISTRY_KEYS.SPLIT_KEY}.png')
        pl.plot_umap(self.adata, slot=umap_slot_key, hue=REGISTRY_KEYS.SPLIT_KEY, output_file=umap_split_o)
        # Plot full UMAP colored by batch label
        umap_batch_o = os.path.join(plt_dir, f'full_umap_{self.original_batch_key}.png')
        pl.plot_umap(self.adata, slot=umap_slot_key, hue=self.original_batch_key, output_file=umap_batch_o)
        # Plot full UMAP colored by batch label
        umap_label_o = os.path.join(plt_dir, f'full_umap_{self.original_label_key}.png')
        pl.plot_umap(self.adata, slot=umap_slot_key, hue=self.original_label_key, output_file=umap_label_o)
        # Plot performance - support correlations
        support_corr_o = os.path.join(plt_dir, f'support_correlations.png')
        pl.plot_performance_support_corr(self.adata.uns[REGISTRY_KEYS.REPORT_KEY], o=support_corr_o, hue=REGISTRY_KEYS.SPLIT_KEY)
        # Plot performance metric over top N predictions in all splits
        top_n_o = os.path.join(plt_dir, f'top_n_{metric}.png')
        pl.plot_top_n_performance(
            self.adata.uns[REGISTRY_KEYS.TOP_N_PREDICTION_KEY],
            output_file=top_n_o,
            metric=metric,
            mean_split='val'
        )
        # Plot individual splits
        self._plot_eval_splits(plt_dir)
        return

    @classmethod
    def load_checkpoint(
        cls,
        model_dir: str,
        adata: AnnData | None,
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
                logging.info(f'Loading model checkpoint {checkpoint_model_p}.')
                checkpoint_model_dir = os.path.dirname(checkpoint_model_p)
                return super().load(checkpoint_model_dir, adata=adata)
        logging.info(f'Could not find model checkpoint(s). Using default "{default_dirname}" directory.')    
        model_state_dir = os.path.join(model_dir, default_dirname)
        return super().load(model_state_dir, adata=adata)
    
    def save(self, keep_emb: bool = True, *args, **kwargs) -> None:
        """Save wrapper to handle external model embeddings"""
        # Handle external model embeddings
        if self.cls_emb is not None:
            # Convert embeddings to numpy arrays
            if keep_emb:
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
        Setup AnnData object for training a JEDVI model.
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
        # Register latent fields of adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        # Create AnnData manager
        adata_manager = EmbAnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
