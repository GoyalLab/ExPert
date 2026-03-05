from typing import Iterable

from src.modules._base import (
    Encoder, 
    GeneEmbeddingEncoder,
    SplitEncoder,
    DecoderSCVI, 
    EmbeddingAligner,
    HierarchicalAligner,
    Classifier,
    ClassEmbedding,
    EfficiencyHead
)
import src.utils.io as io
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS, LOSS_KEYS, PREDICTION_KEYS
import src.utils.embeddings as emb_utils
import src.utils.common as co
from src.utils.augmentations import BatchAugmentation
from src.losses.clip import ClipLoss

from typing import Iterable
import numpy as np

import torch
import torch.nn.functional as F

from scvi.data import _constants
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.module.base import (
    LossOutput,
    auto_move_data,
)
from scvi.module._vae import VAE

from scvi.distributions import (
    NegativeBinomial,
    Normal,
    Poisson,
    ZeroInflatedNegativeBinomial,
)

from collections.abc import Callable
from typing import Literal

from torch.distributions import Distribution
from scvi.model.base import BaseModelClass

import logging
log = logging.getLogger(__name__)


class XPert(VAE):
    """
    Adaption of scVI and scanVI models to predict Perturb-seq perturbations using context- and class-embedding data.
    """
    _cls_loss_strategies = ['ce', 'focal']
    _align_ext_emb_loss_strategies = ['clip']
    _ctx_key = 'ctx'
    _cls_key = 'cls'

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_batch: int,
        n_hidden: int = 256,
        n_latent: int = 128,
        n_layers: int = 2,
        n_latent_local: int | None = None,
        n_latent_global: int | None = None,
        n_shared: int | None = None,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        gene_emb: torch.Tensor | None = None,
        cls_text_dict: dict | None = None,
        cls_weights: torch.Tensor | None = None,
        ctrl_class_idx: int | None = None,
        feat_masks: torch.Tensor | None = None,
        use_adapter: bool = False,
        use_reconstruction_control: bool = False,
        use_kl_control: bool = False,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.2,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        use_cpm: bool = False,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        encode_covariates: bool = False,
        decode_covariates: bool = True,
        decode_ctx_emb: bool = False,
        decode_context_projection: bool = False,
        deeply_inject_covariates: bool = False,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        l1_lambda: float | None = 1e-5,
        l2_lambda: float | None = 1e-3,
        l_mask: str | list[str] | None = None,
        min_kl: float | None = 1.0,
        align_ext_emb_loss_strategy: Literal['kl', 'clip'] | list[str] = 'clip',            # Secondary classifier module (links external embedding)
        reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        non_elbo_reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        use_hierarchical_labels: bool = True,
        encoder_type: Literal['default', 'split', 'gene'] = 'default',
        hierarchy_inference: Literal['embedding', 'library'] = 'embedding',
        use_hierarchical_clip: bool = False,
        use_learnable_hierarchy_temperatures: bool = False,
        module_quantile: float = 0.9,
        pathway_quantile: float = 0.7,
        use_cls_weights: bool = True,
        use_augmentation: bool = False,
        use_decoded_augmentation: bool = False,
        shuffle_context: bool = True,
        link_latents: bool = False,
        automatic_loss_scaling: bool = False,
        model_efficiency: bool = True,
        extra_encoder_kwargs: dict | None = {},
        extra_decoder_kwargs: dict | None = {},
        extra_aligner_kwargs: dict | None = {},
        extra_cls_kwargs: dict | None = {},
        extra_cls_emb_kwargs: dict | None = {},
        extra_aug_kwargs: dict | None = {},
        extra_clip_kwargs: dict = {},
        **vae_kwargs
    ):
        # Initialize base model class
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=False,
            use_observed_lib_size=True,
            var_activation=var_activation,
            **vae_kwargs
        )

        # Setup control index
        self.ctrl_class_idx = ctrl_class_idx
        self.use_reconstruction_control = use_reconstruction_control
        self.use_kl_control = use_kl_control
        # Update labels, we don't predict control as a class
        self.n_labels = n_labels if self.ctrl_class_idx is None else n_labels - 1

        # Setup l-norm params
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.l_mask = l_mask

        # Setup extra batch transformation params
        self.use_cpm = use_cpm
        self.automatic_loss_scaling = automatic_loss_scaling

        # Setup extra basic vae args
        self.min_kl = min_kl

        # Remove control embedding if it exists
        if self.ctrl_class_idx is not None and cls_emb is not None:
            no_ctrl_emb_mask = torch.arange(cls_emb.size(0)) != self.ctrl_class_idx
            cls_emb = cls_emb[no_ctrl_emb_mask]

        # Set reduction metrics
        if reduction not in ['batchmean', 'mean']:
            raise ValueError(f'Invalid reduction for elbo loss metrics: {reduction}, choose either "batchmean", or "mean".')
        if non_elbo_reduction not in ['batchmean', 'mean', 'sum']:
            raise ValueError(f'Invalid reduction for extra loss metrics: {non_elbo_reduction}, choose either "batchmean", or "mean".')
        self.reduction = reduction
        self.non_elbo_reduction = non_elbo_reduction
        
        # Setup data augmentation module
        self.augmentation = BatchAugmentation(**extra_aug_kwargs) if use_augmentation else None
        # Additional batch augmentation via decoded sample
        self.use_decoded_augmentation = use_decoded_augmentation
        self.shuffle_context = shuffle_context
        # Setup class weights (optional)
        self.cls_weights = cls_weights if use_cls_weights else None
        # Whether to model the efficiency score (only available if score is given)
        self.model_efficiency = model_efficiency

        # Setup feature masks
        if feat_masks is not None:
            self.register_buffer('feat_masks', feat_masks)
        else:
            self.feat_masks = None
        
        # Setup external embeddings
        self.ctx_emb = torch.nn.Embedding.from_pretrained(ctx_emb, freeze=True) if ctx_emb is not None else None
        self.use_adapter = use_adapter
        if use_adapter and not link_latents:
            extra_cls_emb_kwargs['n_output'] = n_latent
        # Setup class embedding
        if cls_emb is not None or cls_text_dict is not None:
            self.cls_emb = ClassEmbedding(pretrained_emb=cls_emb, class_texts=cls_text_dict, **extra_cls_emb_kwargs)
            self.n_cls = self.cls_emb.shape[0]
            self.n_cls_dim = self.cls_emb.shape[-1]
        else:
            self.cls_emb = None
            self.n_cls, self.n_cls_dim = 0, 0
        self.cls_sim = torch.nn.Embedding.from_pretrained(cls_sim, freeze=True) if cls_sim is not None  else None
        # Hierarchical clustering parameters
        self.module_quantile = module_quantile
        self.pathway_quantile = pathway_quantile
        self.cls2module = None  # Will be set on first reset_cached_cls_emb
        self.cls2pw = None
        self.module2pw = None
        self.module_weight = None
        self.pathway_weight = None
        # Setup embedding params
        if ctx_emb is not None:
            self.n_ctx, self.n_ctx_dim = ctx_emb.shape
        elif feat_masks is not None:
            self.n_ctx, self.n_ctx_dim = n_batch, n_input
        else:
            self.n_ctx, self.n_ctx_dim = 0, 0
        self.use_joint = False
        # Set latent dimension to text encoder output dim
        if link_latents:
            n_latent = self.n_cls_dim
        # Setup class proxies and hierachical labels
        self.use_hierarchical_labels = use_hierarchical_labels
        self.use_hierarchical_clip = use_hierarchical_clip
        self.hierarchy_inference = hierarchy_inference
        self.gene_names = np.array(list(cls_text_dict.keys())) if cls_text_dict is not None else None
        # Set unseen indices for all layers
        self.unseen_gene_indices = torch.arange(self.n_labels, self.n_cls)
        self.reset_cached_cls_emb(verbose=True, reset_labels=True)
        # Setup learnable temperatures
        self.use_learnable_hierarchy_temperatures = use_learnable_hierarchy_temperatures
        if use_learnable_hierarchy_temperatures:
            self.log_t_gene = torch.nn.Parameter(torch.log(torch.tensor(1/0.1)))
            self.log_t_module = torch.nn.Parameter(torch.log(torch.tensor(1/0.25)))
            self.log_t_pathway = torch.nn.Parameter(torch.log(torch.tensor(1/1.0)))
        # Save if we have unseen embeddings or not
        self.has_unseen_ctx = self.n_ctx > n_batch
        self.has_unseen_cls = self.n_cls > n_labels
        self.n_shared = n_shared if n_shared is not None else n_latent

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'
        # Set covaritates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)

        # ----- Setup encoder module -----
        encoder_cat_list = cat_list if encode_covariates else None
        self.default_encoder_kwargs = {
            'n_input': n_input,
            'n_output': n_latent,
            'n_hidden': n_hidden,
            'n_cat_list': encoder_cat_list,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm_encoder,
            'use_layer_norm': use_layer_norm_encoder,
            'var_activation': var_activation,
            'n_dim_context_emb': self.n_ctx_dim,
            'gene_embedding_init': gene_emb
        }
        # Set base decoder input dimension
        base_decoder_input_dim = n_latent
        
        # Set encoder type
        if encoder_type == 'default':
            encoder_cls = Encoder
        elif encoder_type == 'split':
            encoder_cls = SplitEncoder
            self.default_encoder_kwargs['n_output_global'] = n_latent_global
            self.default_encoder_kwargs['n_output_local'] = n_latent_local
            n_latent = n_latent_local
        elif encoder_type == 'gene':
            encoder_cls = GeneEmbeddingEncoder
        else:
            raise ValueError(f'Unrecognized encoder type "{encoder_type}".')
        self.encoder_type = encoder_type
        extra_encoder_kwargs.update(self.default_encoder_kwargs)
        # Create encoder class
        self.z_encoder = encoder_cls(**extra_encoder_kwargs)

        # ----- Setup decoder module -----
        # Covariate params
        self.decode_covariates = decode_covariates
        self.decode_ctx_emb = decode_ctx_emb
        self.decode_context_projection = decode_context_projection
        # Whether to decode covariates
        n_input_decoder = base_decoder_input_dim + n_continuous_cov * decode_covariates
        # Set decoder input to global if not None
        decoder_cat_list = cat_list if decode_covariates else None
        n_ctx_dim_decoder = None
        # Whether to include context embedding in decoder
        if decode_ctx_emb:
            # Use projected context embedding
            if decode_context_projection:
                n_ctx_dim_decoder = self.n_shared
            # Use raw context embedding
            else:
                n_ctx_dim_decoder = self.n_ctx_dim
        
        # Setup aligner
        if self.use_hierarchical_labels and self.use_hierarchical_clip:
            # Hierachical aligner
            self.aligner = HierarchicalAligner(
                n_latent=n_latent,
                n_emb=self.n_cls_dim,
                n_small_proxies=self.n_pw,
                n_med_proxies=self.n_mod,
                **extra_aligner_kwargs
            )
        else:
            self.aligner = EmbeddingAligner(
                n_input=n_latent,
                shared_projection_dim=self.n_shared,
                class_embed_dim=self.n_cls_dim,
                **extra_aligner_kwargs
            )    
        
        # Init actual decoder
        self.decoder = DecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cat_list=decoder_cat_list,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder, 
            use_layer_norm=use_layer_norm_decoder,
            scale_activation='softmax',
            inject_covariates=deeply_inject_covariates,
            n_dim_context_emb=n_ctx_dim_decoder,
            **extra_decoder_kwargs,
        )
        
        # Setup clip loss module
        extra_clip_kwargs['reduction'] = self.non_elbo_reduction
        self.clip_loss = ClipLoss(**extra_clip_kwargs)

        # Setup an additional classifier on z, purely supervised
        self.classifier = Classifier(
            n_input=n_latent,
            n_labels=self.n_labels,
            **extra_cls_kwargs,
        )
        
        # Setup efficiency head if enabled
        if self.model_efficiency:
            self.efficiency_head = EfficiencyHead(n_latent)
            self.responder_head = EfficiencyHead(n_latent)
       
        # Check loss strategies
        self.set_align_ext_emb_strategies(align_ext_emb_loss_strategy)

        # Check parameter setup
        self._check_setup()

        # ----- Debug -----
        self._step = 0
        
    def _check_setup(self, error_on_fail: bool = False):
        # Error message definition
        def e(msg):
            if error_on_fail:
                raise ValueError(msg)
            else:
                log.warning(msg)
        # TODO: Add compatibiliy checks
        
    def get_device(self):
        # Set own device
        if not hasattr(self, 'parameters'):
            return None
        _device_list = list({p.device for p in self.parameters()})
        if len(_device_list) > 0:
            return _device_list[0]
        return None
    
    def _get_module_by_name(self, name: str):
        module = self
        for part in name.split("."):
            if not hasattr(module, part):
                return None
            module = getattr(module, part)
        return module

    def freeze_module(
        self,
        key: str,
        soft_lr: float | None = None,
        optim: torch.nn.Module | None = None
    ) -> None:

        module = self._get_module_by_name(key)

        if module is None or not isinstance(module, torch.nn.Module):
            log.warning(f'{key} is not a valid module, skipped freeze.')
            return
        # Module already frozen
        if getattr(module, 'frozen', False):
            return

        # HARD FREEZE
        if soft_lr is None or optim is None:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            log.info(f'Successfully froze {key} parameters.')
            module.frozen = True

        # SOFT FREEZE (reduce LR)
        else:
            log.info(f'Successfully soft froze {key} parameters with lr: {soft_lr}.')
            module_params = set(module.parameters())

            for group in optim.param_groups:
                if any(p in module_params for p in group["params"]):
                    group["lr"] = soft_lr
                    
    def unfreeze_module(
        self,
        key: str,
        restore_lr: float | None = None,
        optim: torch.nn.Module | None = None
    ) -> None:

        module = self._get_module_by_name(key)

        if module is None or not isinstance(module, torch.nn.Module):
            log.warning(f'{key} is not a valid module, skipped unfreeze.')
            return

        # re-enable gradients
        module.train()
        for param in module.parameters():
            param.requires_grad = True

        # restore LR if requested
        if restore_lr is not None and optim is not None:
            module_params = set(module.parameters())
            for group in optim.param_groups:
                if any(p in module_params for p in group["params"]):
                    group["lr"] = restore_lr

        log.info(f'Successfully unfroze {key} parameters.')
        module.frozen = False
        
    def reset_cached_cls_emb(self, verbose: bool = False, reset_labels: bool = False):
        """Reset embedding on epoch start"""
        # Recompute gene embeddings
        self.cached_cls_emb = self.cls_emb()
        # Reset hierarchical labels
        if self.use_hierarchical_labels:
            if reset_labels:
                self._set_hierarchical_labels(self.cached_cls_emb, verbose=verbose)
        
    def _set_hierarchical_labels(
        self,
        cls_emb: torch.Tensor | None,
        verbose: bool = False
    ):
        """Set module and pathway labels with adaptive thresholding"""
        from src.utils.annotations import GeneAnnotation
        n = self.n_cls
        
        # Get annotation-based gene information instead of provided class embedding
        if self.hierarchy_inference == 'library' or cls_emb is None:
            # Use gene names to get annotation-based clustering
            gene_names = self.gene_names
            gene_emb = None
        elif self.hierarchy_inference == 'embedding':
            # Use given gene embedding as base for clustering
            gene_names = None
            gene_emb = cls_emb
        else:
            raise ValueError(f'Unknown argument {self.hierarchy_inference} for self.hierarchy_inference.')
        
        # Set hierarchy based on class embedding similarities
        gh_kwargs = {}
        if not self.use_hierarchical_clip:
            gh_kwargs['method'] = None
        
        # Create gene annotations
        ga = GeneAnnotation(gene_names=gene_names, gene_emb=gene_emb, device=self.get_device(), verbose=verbose, **gh_kwargs)
        self.gene_hierarchy = ga
        self.cls2pw = ga.get_cls2pw()
        self.cls2module = ga.get_cls2module()
        self.module2pw = ga.get_module2pw()
        # Get miscellanious pathway and module indices
        self.misc_pw_idx = ga.misc_pw_idx
        self.misc_mod_idx = ga.misc_mod_idx
        # Set embedding edge weights and indices
        self.laplacian_graph = ga.laplacian_graph
        
        # Skip hierarchical assignments
        if not self.use_hierarchical_clip:
            return
            
        # Get modules/pathways for observed genes
        observed_modules = set(self.cls2module[:self.n_labels].tolist())
        observed_pathways = set(self.cls2pw[:self.n_labels].tolist())
        self.n_mod = len(set(self.cls2module.tolist()))
        self.n_pw = len(set(self.cls2pw.tolist()))
        self.n_observed_mod = len(observed_modules)
        self.n_observed_pw = len(observed_pathways)
        self.observed_pathways = sorted(observed_pathways)
        self.observed_modules = sorted(observed_modules)
        # Get all unique modules/pathways
        all_modules = set(self.cls2module.tolist())
        all_pathways = set(self.cls2pw.tolist())
        
        # Unseen = all - observed
        self.unseen_module_indices = torch.tensor(
            sorted(all_modules - observed_modules),
            dtype=torch.long,
            device=self.cls2module.device
        )
        
        self.unseen_pathway_indices = torch.tensor(
            sorted(all_pathways - observed_pathways),
            dtype=torch.long,
            device=self.cls2pw.device
        )
        
        # Build module membership for faster predictions
        n_modules = self.cls2module.max().item() + 1
        n_pathways = self.cls2pw.max().item() + 1
        
        # pathway_children[p] = list of fine gene indices in pathway p
        self.pw2module = {}
        for gene_id, pw_id in enumerate(self.cls2pw):
            module_id = self.cls2module[gene_id].item()
            self.pw2module.setdefault(int(pw_id), set()).add(int(module_id))

        # convert sets to lists
        self.pw2module = {k: list(v) for k, v in self.pw2module.items()}


        # module_children[m] = list of pathway indices in module m
        self.module_children = {
            m: torch.where(self.cls2module == m)[0].tolist()
            for m in range(n_modules)
        }
        
        # Module membership: (n_modules, n_genes)
        self.module_membership = torch.zeros(
            n_modules, n, 
            device=self.cls2module.device
        )
        for g_idx in range(n):
            m_idx = self.cls2module[g_idx].item()
            self.module_membership[m_idx, g_idx] = 1.0
        
        # Pathway membership: (n_pathways, n_genes)
        self.pathway_membership = torch.zeros(
            n_pathways, n,
            device=self.cls2pw.device
        )
        for g_idx in range(n):
            p_idx = self.cls2pw[g_idx].item()
            self.pathway_membership[p_idx, g_idx] = 1.0
            
        # Build hierachy masks
        self._build_dense_masks()
        
        # Update aggregated embeddings
        self.cached_pw_emb = self._get_pathway_embeddings(self.cached_cls_emb)
        self.cached_mod_emb = self._get_module_embeddings(self.cached_cls_emb)
        
        if verbose:
            log.info(f"Hierarchical labels computed:")
            log.info(f"  Total genes: {n}")
            log.info(f"     Observed genes: {self.n_labels}")
            log.info(f"     Unseen genes: {n - self.n_labels}")
            log.info(f"  Total modules: {len(all_modules)}")
            log.info(f"     Observed modules: {len(observed_modules)}")
            log.info(f"     Unseen modules: {len(self.unseen_module_indices)}")
            log.info(f"  Total pathways: {len(all_pathways)}")
            log.info(f"     Observed pathways: {len(observed_pathways)}")
            log.info(f"     Unseen pathways: {len(self.unseen_pathway_indices)}")

    def _get_module_embeddings(self, cls2z: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Aggregate gene embeddings into module embeddings"""
        if cls2z is None:
            return None
        cls2module = self.cls2module
        if indices is not None:
            cls2module = cls2module[indices]
        n_modules = cls2module.unique().shape[0]
        module_embs = torch.zeros(n_modules, cls2z.size(1), device=cls2z.device)
        
        for i in range(n_modules):
            mask = (cls2module == i)
            if mask.any():
                module_embs[i] = cls2z[mask].mean(dim=0)
        
        return F.normalize(module_embs, dim=-1)

    def _get_pathway_embeddings(self, cls2z: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Aggregate gene embeddings into pathway embeddings"""
        if cls2z is None:
            return None
        cls2pw = self.cls2pw
        if indices is not None:
            cls2pw = cls2pw[indices]
        n_pathways = cls2pw.unique().shape[0]
        pathway_embs = torch.zeros(n_pathways, cls2z.size(1), device=cls2z.device)
        
        for i in range(n_pathways):
            mask = (cls2pw == i)
            if mask.any():
                pathway_embs[i] = cls2z[mask].mean(dim=0)
        
        return F.normalize(pathway_embs, dim=-1)
    
    def _get_unseen_label_idx(self):
        return getattr(self, 'unseen_gene_indices', None)
    
    def _get_unseen_module_idx(self):
        return getattr(self, 'unseen_module_indices')
    
    def _get_unseen_pathway_idx(self):
        return getattr(self, 'unseen_pathway_indices')
    
    def _build_dense_masks(self):
        # number of modules, pathways, genes
        C_mod = int(self.cls2module.max()) + 1
        C_pw  = int(self.cls2pw.max()) + 1
        C_g   = len(self.cls2module)
        device = self.cls2module.device

        # -------- pathway → module mask (for masking) --------
        self.pw2module_mask = torch.zeros(C_pw, C_mod, dtype=torch.bool, device=device)
        for pw in range(C_pw):
            modules = self.pw2module[pw]           # list[int]
            self.pw2module_mask[pw, modules] = True

        # -------- module → gene mask (for masking) --------
        self.module2gene_mask = torch.zeros(C_mod, C_g, dtype=torch.bool, device=device)
        for mod in range(C_mod):
            genes = self.module_children[mod]      # list[int]
            self.module2gene_mask[mod, genes] = True

        # -------- gene → module aggregation (G2M) --------
        G2M = torch.zeros(C_g, C_mod, device=device)
        for g, mod in enumerate(self.cls2module.tolist()):
            G2M[g, mod] = 1.

        # Normalize (mean of child genes)
        col_sums = G2M.sum(dim=0, keepdim=True)
        col_sums[col_sums == 0] = 1.
        G2M = G2M / col_sums

        # -------- module → pathway aggregation (M2P) --------
        M2P = torch.zeros(C_mod, C_pw, device=device)
        for pw in range(C_pw):
            mods = self.pw2module[pw]              # list of modules under this pathway
            M2P[mods, pw] = 1.

        # Normalize (mean of child modules)
        col_sums = M2P.sum(dim=0, keepdim=True)
        col_sums[col_sums == 0] = 1.
        M2P = M2P / col_sums

        # Store
        self.G2M = G2M
        self.M2P = M2P

    def draw_module(self):
        return
        from torchview import draw_graph

        graph = draw_graph(self, input_size=(self), expand_nested=True)
        graph.visual_graph.render("lightning_graph", format="png")

    def compute_semantic_embedding_targets(
        self,
        emb: torch.Tensor,
        use_gpu: bool = True,
        batch_size: int = 512,
        batched: bool = False,
    ) -> torch.Tensor:
        """
        Compute class similarity matrix for semantic label smoothing in batches.
        emb: (N, D)
        returns: (N, N) normalized similarity matrix.
        """
        device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
        emb = F.normalize(emb.to(device), dim=-1)
        # Do simple MM if tensor is not too large
        if not batched:
            return emb @ emb.T
        
        # Do batched version
        N, D = emb.shape
        targets = torch.zeros((N, N), device=device, dtype=emb.dtype)

        for i in range(0, N, batch_size):
            j_end = min(i + batch_size, N)
            # Compute cosine similarity for this batch vs all embeddings
            batch = emb[i:j_end]                       # (B, D)
            targets[i:j_end] = batch @ emb.T                       # fill in chunk

        return targets

    def set_align_ext_emb_strategies(self, align_ext_emb_loss_strategy) -> None:
        """Set alignment strategies."""
        if align_ext_emb_loss_strategy is None:
            self.align_ext_emb_loss_strategies = None
        # Make sure the resulting type is a list
        self.align_ext_emb_loss_strategies: list[str] = align_ext_emb_loss_strategy if isinstance(align_ext_emb_loss_strategy, list) else [align_ext_emb_loss_strategy]
        # Check for allowed options
        for align_strategy in self.align_ext_emb_loss_strategies:
            if align_strategy not in self._align_ext_emb_loss_strategies:
                raise ValueError(f'Unrecognized alignment strategy: {align_strategy}, choose one of {self._align_ext_emb_loss_strategies}')
        
    @torch.no_grad()
    def _get_decoded_augmentation(
        self,
        tensors: dict[str, torch.Tensor],
        incl_x: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Generate decoded batch as augmentation."""
        # Run full VAE forward pass with no gradient
        _tensors = {k: v for k, v in tensors.items()}
        x = _tensors[MODULE_KEYS.X_KEY]
        B = x.size(0)
        # Inference
        inference_outputs = self._regular_inference(**tensors)
        generative_input = self._get_generative_input(_tensors, inference_outputs)
        # Randomize context labels for generative part
        if self.shuffle_context:
            # Get shape of current label
            label_shape = _tensors[MODULE_KEYS.BATCH_INDEX_KEY].shape
            # Pick random context labels
            generative_input[MODULE_KEYS.BATCH_INDEX_KEY] = torch.randint(0, self.n_ctx, label_shape, device=x.device, dtype=torch.long)
            # Set batch variables
            _tensors[MODULE_KEYS.BATCH_INDEX_KEY] = generative_input[MODULE_KEYS.BATCH_INDEX_KEY]
        generative_output = self.generative(**generative_input)
        # Get reconstructed X from output
        px: Distribution = generative_output[MODULE_KEYS.PX_KEY]
        # Overwrite X key
        _tensors[MODULE_KEYS.X_KEY] = px.sample()
        # Optionally concatenate original + augmented along batch dimension
        if incl_x:
            out = {}
            for k, v in tensors.items():
                v2 = _tensors.get(k, None)
                # only cat batch-aligned tensors
                if torch.is_tensor(v) and torch.is_tensor(v2) and v.shape == v2.shape:
                    out[k] = torch.cat([v, v2], dim=0)
                else:
                    # keep original as-is
                    out[k] = v
            return out
        return _tensors
        
    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        
        # Collect batch data
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        label = tensors[REGISTRY_KEYS.LABELS_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        # Construct batch from data
        o = {
            MODULE_KEYS.X_KEY: x,
            MODULE_KEYS.BATCH_INDEX_KEY: batch_index,
            MODULE_KEYS.LABEL_KEY: label,
            MODULE_KEYS.G_EMB_KEY: tensors.get(REGISTRY_KEYS.GENE_EMB_KEY, None),
            MODULE_KEYS.CONT_COVS_KEY: cont_covs,
            MODULE_KEYS.CAT_COVS_KEY: cat_covs
        }
        return o
    
    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        label: torch.Tensor,
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
        **kwargs
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        # Determine library size
        if self.use_observed_lib_size:
            library = torch.log(x_.sum(1)).unsqueeze(1)
        # Apply CMP normalization to x
        if self.use_cpm:
            x_ = x_ / library * 1e6
        # Apply log1p transformation to input batch
        if self.log_variational:
            x_ = torch.log1p(x_)
        # Use feature masks if given and enabled
        if self.feat_masks is not None:
            feat_mask = self.feat_masks[batch_index.flatten()]
        else:
            feat_mask = None

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # Add context embedding
        ctx_emb = self.ctx_emb.weight if self.ctx_emb is not None else None
        # Perform forward pass through encoder
        inference_out: dict[str, torch.Tensor] = self.z_encoder(
            encoder_input, *categorical_input, 
            g=g, feature_masks=feat_mask, 
            ctx_label=batch_index, context_emb=ctx_emb
        )
        # Unpack encoder output
        qz = inference_out[MODULE_KEYS.QZ_KEY]
        z = inference_out[MODULE_KEYS.Z_KEY]
        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == 'embedding':
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        # Construct output object
        o = {
            MODULE_KEYS.X_KEY: x,
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
            MODULE_KEYS.LABEL_KEY: label,
            MODULE_KEYS.BATCH_INDEX_KEY: batch_index,
            MODULE_KEYS.CONT_COVS_KEY: cont_covs,
            MODULE_KEYS.CAT_COVS_KEY: cat_covs,
        }
        # Add local latent
        if self.encoder_type == 'split':
            o.update({
                # Local
                'qzl': inference_out.get('qzl'),
                'zl': inference_out.get('zl'),
                'x_norm': inference_out.get('x_norm'),
                'x_base': inference_out.get('x_base'),
                'x_res': inference_out.get('x_res'),
            })
        return o
    
    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(MODULE_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        # Return generative data
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY],
            MODULE_KEYS.Y_KEY: inference_outputs[MODULE_KEYS.LABEL_KEY],
            MODULE_KEYS.CONT_COVS_KEY: inference_outputs.get(MODULE_KEYS.CONT_COVS_KEY),
            MODULE_KEYS.CAT_COVS_KEY: inference_outputs.get(MODULE_KEYS.CAT_COVS_KEY),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
            MODULE_KEYS.CTX_PROJ_KEY: inference_outputs.get(MODULE_KEYS.CTX_PROJ_KEY),
        }
    
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        ctx_proj: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library
        # Add context embedding to decoder
        ctx_emb = None
        if self.decode_ctx_emb:
            # Use feature masks if given and enabled
            if self.feat_masks is not None:
                ctx_emb = self.feat_masks[batch_index.flatten()]
            elif self.ctx_emb is not None:
                # Get inflated context embeddings for batch (either projected or raw)
                if self.decode_context_projection:
                    ctx_emb = ctx_proj[batch_index.squeeze(-1)]
                else:
                    ctx_emb = self.ctx_emb(batch_index).reshape(batch_index.shape[0], -1)
            
        # Perform decoder forward pass
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            *categorical_input,
            context_emb=ctx_emb
        )

        if self.dispersion == "gene-label":
            px_r = linear(
                F.one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(px_rate, px_r, normal_mu=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }
    
    @auto_move_data
    @torch.no_grad()
    def classify(
        self,
        x: torch.Tensor | None,
        batch_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        inference_outputs: dict[str, torch.Tensor] | None = None,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        local_predictions: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import src.performance as pf
        """Forward pass through the encoder and classifier.

        Parameters
        ----------
        x
            Tensor of shape ``(n_obs, n_vars)``.
        batch_index
            Tensor of shape ``(n_obs,)`` denoting batch indices.
        cont_covs
            Tensor of shape ``(n_obs, n_continuous_covariates)``.
        cat_covs
            Tensor of shape ``(n_obs, n_categorical_covariates)``.
        use_posterior_mean
            Whether to use the posterior mean of the latent distribution for
            classification.
      
        Returns
        -------
        Tensor of shape ``(n_obs, n_labels)`` denoting logit scores per label.
        Before v1.1, this method by default returned probabilities per label,
        see #2301 for more details.
        """
        # Try caching inference and recalculate if missing
        if inference_outputs is None:
            inference_outputs = self.inference(
                x, 
                batch_index, 
                label, 
                g, 
                cont_covs, 
                cat_covs,
            )
        # Get inference outputs
        qz_key = 'qzl' if 'qzl' in inference_outputs else MODULE_KEYS.QZ_KEY
        z_key = 'zl' if 'zl' in inference_outputs else MODULE_KEYS.Z_KEY
        qz = inference_outputs[qz_key]
        z = inference_outputs[z_key]
        # Aligner forward pass using either qz mean or sampled z
        _z = qz.loc if use_posterior_mean else z
        
        # Optionally use different embeddings, fall back to internals if none are given
        if ctx_emb is None and self.ctx_emb is not None:
            ctx_emb = self.ctx_emb.weight
        if cls_emb is None:
            cls_emb = self.cached_cls_emb
        # Subset class embedding to local (trained) classes
        if local_predictions:
            cls_emb = cls_emb[:self.n_labels]

        # Use regular alignment
        align_out: dict[str, torch.Tensor] = self.aligner(
            x=_z, cls_emb=cls_emb,
            return_logits=False
        )
        # Get aligned spaces
        z2c = align_out.get(MODULE_KEYS.Z_SHARED_KEY)
        c2z = align_out.get(MODULE_KEYS.CLS_PROJ_KEY)
        # Get cell-wise T predictions if enabled
        if self.model_efficiency and hasattr(self, 'last_t_gamma'):
            eff_hat = self.efficiency_head(z2c).clamp(max=self.eff_max) / self.eff_max
            T_i = self.aligner.temperature * (1 + self.last_t_gamma * (1 - eff_hat))
            T_i = T_i.clamp(min=0.07, max=1.0).unsqueeze(-1)
        else:
            T_i = self.aligner.temperature
        # Get logits
        align_out[MODULE_KEYS.CLS_LOGITS_KEY] = self.clip_loss._logits_z2c(z2c, c2z, T=T_i)
        # Add response probability to alignment output
        align_out['response_prob'] = pf.compute_response_probability(align_out[MODULE_KEYS.CLS_LOGITS_KEY])
        
        # Add alignment output to inference
        inference_outputs.update(align_out)
        return inference_outputs

    def _reduce_loss(self, loss: torch.Tensor, reduction: str) -> torch.Tensor:
        # Check if loss is a tensor
        if loss is None or not isinstance(loss, torch.Tensor):
            return loss
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'batchmean':
            return loss.sum(-1).mean()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')

    def _grad_reversed_entropy_loss(self, logits: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Maximize entropy via gradient reversal."""
        logits_rev = co.grad_reverse(logits, lam)
        p = torch.softmax(logits_rev, dim=-1)
        return (p * p.log()).sum(dim=-1).mean()

    def get_augmented_alignment(
        self, 
        inference_outputs: dict[str, torch.Tensor],
        cls_emb: torch.Tensor,
        g: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        T: float | None = None,
    ):
        # Get augmentated batch and do forward pass through (local) encoder
        augmented_inputs = self.augmentation(inference_outputs)
        # Get model inference
        inference_outputs = self.inference(**augmented_inputs, g=g)
        # Align augmented batch
        zk = 'zl' if 'zl' in inference_outputs else MODULE_KEYS.Z_KEY
        qzk = 'qzl' if 'qzl' in inference_outputs else MODULE_KEYS.QZ_KEY
        z: torch.Tensor = inference_outputs[qzk].loc if use_posterior_mean else inference_outputs[zk]
        aug_align_out: dict[str, torch.Tensor] = self.aligner(z, cls_emb, T=T, return_logits=False)
        aug_align_out.update(augmented_inputs)
        return aug_align_out

    def _scale_weight_to_elbo(self, L_elbo: torch.Tensor, L_new: torch.Tensor, target_ratio: float = 0.2, min: float = 1e-5, max: float = 100.0):
        # compute grad norms wrt shared parameters (e.g., encoder + z-proj)
        params = [p for p in self.z_encoder.parameters() if p.requires_grad]
        g_elbo = torch.autograd.grad(L_elbo, params, retain_graph=True, create_graph=False, allow_unused=True)
        g_new = torch.autograd.grad(L_new, params, retain_graph=True, create_graph=False, allow_unused=True)
        # Remove None types
        g_elbo = [gi for gi in g_elbo if gi is not None]
        g_new = [gi for gi in g_new if gi is not None]
        # Normalize
        norm_elbo = torch.sqrt(sum((gi.detach()**2).sum() for gi in g_elbo)).clamp_min(1e-8)
        norm_new = torch.sqrt(sum((gi.detach()**2).sum() for gi in g_new)).clamp_min(1e-8)
        # Get weight
        w = (target_ratio * norm_elbo / norm_new).clamp(min, max)
        return w
    
    # Soft responder target using sigmoid around threshold
    def soft_responder_target(self, scores, threshold, steepness: float = 2.0):
        """Smooth binary target, 0.5 at threshold."""
        return torch.sigmoid(steepness * (scores.abs() - threshold)).flatten()
    
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        adv_ctx_weight: float = 0.0,
        ctx_weight: float = 0.0,
        cls_align_weight: float = 1.0,
        l_kl_weight: float = 0.0,
        decor_weight: float = 0.0,
        use_posterior_mean: bool = True,
        local_predictions: bool = True,
        T_align: float | None = None,
        clip_k: int | None = None,
        aug_weight: float = 0.0,
        scale_rl_by_features: bool = False,
        lambda_eff: float = 0.1,
        T_gamma: float = 0.0,
        **kwargs,
    ) -> LossOutput:
        # ---- DEBUG ----
        self._step = self._step + 1
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # Unpack input tensor
        x: torch.Tensor = inference_outputs[MODULE_KEYS.X_KEY]
        ctx: torch.Tensor = inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY]
        cls: torch.Tensor = inference_outputs[MODULE_KEYS.LABEL_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]
        g: torch.Tensor | None = tensors.get(REGISTRY_KEYS.GENE_EMB_KEY, None)
        weights: torch.Tensor | None = tensors.get(REGISTRY_KEYS.CLS_EFF_KEY, None)

        # Compute basic kl divergence between prior and posterior x distributions
        kl_divergence_z_mat = kl_divergence(qz, pz)
        # Calculate reconstruction loss over batch and all features
        reconst_loss_mat = -px.log_prob(x)
        
        # Apply feature masking to loss        
        if self.feat_masks is not None:
            mask_batch = self.feat_masks[ctx.flatten()]
            reconst_loss_mat = reconst_loss_mat * mask_batch
            valid_feat_count = mask_batch.sum(-1).clamp(min=1.0)
        else:
            mask_batch = None
            valid_feat_count = None
        
        # Clamp KL to a minimum if option is provided
        if self.min_kl is not None:
            kl_divergence_z_mat = torch.clamp(kl_divergence_z_mat, min=self.min_kl)
        # Aggregate elbo losses over latent dimensions / input features, will get batch normalized internally
        if self.reduction == 'batchmean':
            kl_divergence_z = kl_divergence_z_mat.sum(-1) # (b,)
            kl_div_per_latent = kl_divergence_z_mat.sum(0) # (l,)
            reconst_loss = reconst_loss_mat.sum(-1)
        elif self.reduction == 'mean':
            kl_divergence_z = kl_divergence_z_mat.mean(-1) # (b,)
            kl_div_per_latent = kl_divergence_z_mat.mean(0) # (l,)
            # Normalize by valid features per cell
            if self.feat_masks is not None:
                reconst_loss = (reconst_loss_mat.sum(-1) / valid_feat_count)
            else:
                reconst_loss = reconst_loss_mat.mean(-1)
            if scale_rl_by_features:
                reconst_loss = reconst_loss * reconst_loss_mat.size(1)
        else:
            raise ValueError(f'reduction has to be either "batchmean" or "mean", got {self.reduction}')

        # Use full batch by default
        n_obs_minibatch = x.shape[0]
        # Use full batch as default
        ctrl_mask = None
        n_obs_minibatch = x.shape[0]
        # Handle control cells for elbo loss
        if self.ctrl_class_idx is not None:
            # Calculate control mask
            ctrl_mask = (cls == self.ctrl_class_idx).reshape(-1)
            # Check if there are any control cells in the batch
            if ctrl_mask.sum() > 0:
                # Calculate fraction of control cells in batch
                ctrl_frac = x.shape[0] / max(ctrl_mask.sum(), 1)    # avoid div/0
                # Divide elbo sum by class cells only
                n_obs_minibatch = (~ctrl_mask).sum()
                # Disregard elbo for control cells
                if not self.use_reconstruction_control and not self.use_kl_control:
                    reconst_loss = reconst_loss * ~ctrl_mask
                    kl_divergence_z = kl_divergence_z * ~ctrl_mask
                elif not self.use_reconstruction_control:
                    reconst_loss = reconst_loss * ~ctrl_mask
                    kl_weight = kl_weight * ctrl_frac
                elif not self.use_kl_control:
                    kl_divergence_z = kl_divergence_z * ~ctrl_mask
                    rl_weight = rl_weight * ctrl_frac
                else:
                    # Use full batch
                    n_obs_minibatch = x.shape[0]
        
        # Weighted reconstruction and KL
        weighted_reconst_loss = (rl_weight * reconst_loss).mean()
        weighted_kl_local = (kl_weight * kl_divergence_z).mean()

        # Save reconstruction losses
        kl_locals = {MODULE_KEYS.KL_Z_KEY: kl_divergence_z}
        # Batch normalize elbo loss of reconstruction + kl --> batchmean reduction
        L_elbo = weighted_reconst_loss + weighted_kl_local
        total_loss = L_elbo

        # Create extra metric container
        extra_metrics = {MODULE_KEYS.KL_Z_PER_LATENT_KEY: kl_div_per_latent}

        # Collect inital loss and extra parameters
        lo_kwargs = {
            'reconstruction_loss': reconst_loss,
            'kl_local': kl_locals,
            'true_labels': cls,
            'n_obs_minibatch': n_obs_minibatch
        }
        # Monitor latent space ranks
        extra_metrics['z_rank'] = torch.linalg.matrix_rank(z).float().mean()

        # Collect extra outputs
        extra_outputs = {}
        
        # Re-format labels
        ctx = ctx.flatten()
        cls = cls.flatten()
        
        # Optional context classifier on z
        if ctx_weight > 0:
            ctx_z = qz.loc if use_posterior_mean else z
            ctx_loss, ctx_pred, ctx_labels = self.context_loss(ctx_z, ctx)
            total_loss = total_loss + ctx_weight * ctx_loss
            extra_metrics['L_ctx'] = ctx_loss
            extra_outputs['ctx_pred'] = ctx_pred

        # Set empty classification logits
        logits = None
        extra_outputs['ctx_labels'] = ctx

        # Use local latent space if available, else fall back to shared latent
        if 'qzl' in inference_outputs:
            _qz = inference_outputs['qzl']
            z_ = inference_outputs['zl']
            has_local = True
        else:
            _qz = qz
            z_ = z
            has_local = False
        # Base classification / clip losses on posterior mean or sampled latent space
        _z = _qz.loc if use_posterior_mean else z_
        
        # Split latent regularizations
        if has_local:
            # Add KL-loss on local latent distribution
            if l_kl_weight > 0:
                kl_divergence_zl_mat = kl_divergence(inference_outputs['qzl'], Normal(torch.zeros_like(z), torch.ones_like(z)))
                # Aggregate elbo losses over latent dimensions / input features, will get batch normalized internally
                kl_divergence_zl = self._reduce_loss(kl_divergence_zl_mat, self.non_elbo_reduction)
                total_loss = total_loss + l_kl_weight * kl_divergence_zl
                extra_metrics['kl_zl_local'] = kl_divergence_zl
            # Add decorrelation regularizer on global vs local
            if decor_weight > 0:
                reg_decorr = self.z_encoder.decorrelation_reg(
                    z_g=qz.loc if use_posterior_mean else z,
                    z_l=_z
                )
                reg_decorr = self._reduce_loss(reg_decorr, self.non_elbo_reduction)
                total_loss = total_loss + decor_weight * reg_decorr
                extra_metrics['L_decorr'] = reg_decorr

        # Get class embedding transformer output
        cls_emb = self.cached_cls_emb
        # Do alignment forward pass and calculate losses
        alignment_output = self.aligner(
            x=_z, cls_emb=cls_emb, return_logits=False
        )
        # Extract shared latent space
        z_shared = alignment_output.get(MODULE_KEYS.Z_SHARED_KEY)
        # Extract the embedding projections to shared space
        cls2z = alignment_output.get(MODULE_KEYS.CLS_PROJ_KEY)
        # Add projected embedding representations to extra outputs
        extra_outputs[MODULE_KEYS.Z_SHARED_KEY] = F.normalize(z_shared, dim=-1).detach()
        extra_outputs[MODULE_KEYS.CLS_PROJ_KEY] = F.normalize(cls2z, dim=-1).detach()

        # Choose alignment temperature
        T_align = T_align if T_align is not None else self.aligner.temperature
        # Predict perturbation efficiency per cell
        if self.model_efficiency and weights is not None and lambda_eff > 0:
            weights = weights.flatten()
            # Predict responders
            responder_logits = self.responder_head(z_shared)
            responder_target = self.soft_responder_target(weights, threshold=2.0)
            responder_loss = F.binary_cross_entropy_with_logits(
                responder_logits, responder_target, reduction='mean'
            )
            extra_metrics['responder_loss'] = responder_loss
            total_loss = total_loss + lambda_eff * responder_loss
            # Responder gate
            gate = torch.sigmoid(responder_logits.detach()).squeeze(-1)
            # Efficiency regression 
            eff_hat = self.efficiency_head(z_shared)
            eff_loss = F.mse_loss(eff_hat, weights, reduction='mean')
            total_loss = total_loss + lambda_eff * eff_loss
            extra_metrics['eff_loss'] = eff_loss

            # Temperature modulation — gate controls inclusion, eff controls sharpness
            eff_max = weights.max()
            if hasattr(self, 'last_eff_max'):
                self.eff_max = torch.max(eff_max, self.last_eff_max)
            else:
                self.eff_max = eff_max
            eff_norm = eff_hat.abs().detach().squeeze(-1).clamp(0, self.eff_max) / self.eff_max
            T_i = T_align * (1 + T_gamma * (1 - eff_norm))
            T_i = T_i.clamp(min=0.07, max=1.0).unsqueeze(-1)
            # Log data distributions
            extra_outputs[REGISTRY_KEYS.CLS_EFF_KEY] = weights
            extra_outputs['eff_hat'] = eff_hat
            # Log T distribution
            extra_outputs['T_align_dist'] = T_i
            # Keep track of last model T gamma for inference
            self.last_t_gamma = T_gamma
            # Pass gate as weights
            clip_weights = gate
            extra_outputs['clip/weights'] = eff_hat
            extra_outputs['responder_prob'] = gate
            extra_outputs['responder_prob_target'] = responder_target
        else:
            # Use actual weights for clip
            clip_weights = None
            T_i = None
        # Log global temperature
        extra_metrics['T_align'] = T_align
        
        # Add alignment loss if non-zero weight
        if io.non_zero(cls_align_weight):
            # Adversial context loss
            if adv_ctx_weight > 0:
                L_adv_ctx, ctx_pred, ctx_labels = self.adversarial_context_loss(z_shared, ctx)
                total_loss = total_loss + adv_ctx_weight * L_adv_ctx
                extra_metrics['L_adv_ctx'] = L_adv_ctx
                extra_outputs['adv_ctx_pred'] = ctx_pred
                extra_outputs['adv_ctx_labels'] = ctx_labels
            
            # Get augmented aligment output
            if self.training and self.augmentation is not None and aug_weight > 0:
                aug_align_out = self.get_augmented_alignment(
                    inference_outputs, cls_emb=cls_emb, T=T_align, g=g, use_posterior_mean=use_posterior_mean,
                )
                n_aug = aug_align_out[MODULE_KEYS.Z_SHARED_KEY].size(0)
                # Set initial weights if None
                clip_weights = torch.ones_like(cls, device=cls.device)
                # Update clip loss inputs
                z_shared = torch.cat(
                    [z_shared, aug_align_out[MODULE_KEYS.Z_SHARED_KEY]], dim=0
                )
                cls = torch.cat((cls, aug_align_out[MODULE_KEYS.LABEL_KEY]), dim=0)
                ctx = torch.cat((ctx, aug_align_out[MODULE_KEYS.BATCH_INDEX_KEY]), dim=0)
                # Update loss kw args
                lo_kwargs['true_labels'] = cls
                # Update weights
                clip_weights = torch.cat([clip_weights, torch.ones(n_aug, device=cls.device) * aug_weight], dim=0)
            
            # Calculate class similarities
            if local_predictions:
                local_targets = torch.arange(self.n_labels, device=cls2z.device)
                # Add null targets to logits
                if self.cls_emb.use_null_proxy:
                    # Determine targets
                    all_targets = torch.arange(cls2z.size(0), device=cls2z.device)
                    null_targets = all_targets[all_targets >= self.n_cls]
                    local_targets = torch.cat([local_targets, null_targets])
                # Subset to local targets
                _cls_emb: torch.Tensor = cls2z[local_targets]
                n_cls = self.n_labels
            else:
                # Use all available targets
                _cls_emb = cls2z
                n_cls = self.n_cls
                
            # Calculate individual alignment losses
            alignment_losses: dict = self.clip_loss(
                z=z_shared,
                y=cls.flatten(),
                ctx=ctx.flatten(),
                emb=_cls_emb,
                T=T_align,
                k=clip_k,
                weights=clip_weights,
                T_i=T_i,
            )
            # Add margins to extra outputs
            extra_outputs['clip/margin'] = alignment_losses.pop('clip/margin', None)
            # Get clip logits
            _T = T_i if T_i is not None else T_align
            logits = self.clip_loss._logits_z2c(z_shared, _cls_emb, T=_T)

            # Add combined alignment loss to final loss
            align_loss = alignment_losses.pop(LOSS_KEYS.LOSS)
            
            # Add some regularizers to class embedding
            # TODO: add scheduled weights
            emb_prot_div_reg = self.cls_emb.prototype_div_reg(_cls_emb)
            extra_metrics['emb/prot_div_reg'] = emb_prot_div_reg
            align_loss = align_loss + 0.01 * emb_prot_div_reg
            
            # Add null prior loss
            if self.cls_emb.use_null_proxy:
                # Determine targets
                all_local_targets = torch.arange(_cls_emb.size(0), device=cls2z.device)
                null_local_targets = all_local_targets[all_local_targets >= n_cls]
                L_emb_null_frac = self.clip_loss.null_fraction_loss(logits, null_local_targets)
                extra_metrics['emb/L_null_frac'] = L_emb_null_frac
                total_loss = total_loss + 0.1 * L_emb_null_frac
            
            # Log individual alignment losses
            extra_metrics.update(alignment_losses)
                        
            # Scale weights based on L_elbo
            if self.training:
                # Scale clip according to reconstruction loss gradient
                if self.automatic_loss_scaling and (self._step % 10 == 0 or not hasattr(self, 'cls_align_weight')):
                    self.cls_align_weight = self._scale_weight_to_elbo(reconst_loss.mean(), align_loss, target_ratio=cls_align_weight)
                # Set to given weight
                else:
                    self.cls_align_weight = cls_align_weight
                    
                # Log clip weight
                extra_metrics['clip/weight'] = self.cls_align_weight
            
            # Add alignment loss to total loss
            total_loss = total_loss + self.cls_align_weight * align_loss
            extra_metrics[LOSS_KEYS.ALIGN_LOSS] = align_loss
        else:
            alignment_output = None
            
        # Only return predictions of observable classes
        if logits is not None and PREDICTION_KEYS.PREDICTION_KEY not in extra_outputs:
            # Add class predictions to extra output
            extra_outputs[PREDICTION_KEYS.PREDICTION_KEY] = logits.argmax(dim=-1).flatten().detach()
        # Add logits to extra output
        if logits is not None:
            extra_outputs[MODULE_KEYS.CLS_LOGITS_KEY] = logits.detach()
        # Add extra outputs to extra metrics
        extra_metrics['extra_outputs'] = extra_outputs
        # Add extra metrics to loss output
        lo_kwargs['extra_metrics'] = extra_metrics
        # Set total loss
        lo_kwargs[LOSS_KEYS.LOSS] = total_loss
        return LossOutput(**lo_kwargs)

    def on_load(self, model: BaseModelClass):
        """Sync model adata manager on load"""
        manager = model.get_anndata_manager(model.adata, required=True)
        source_version = manager._source_registry[_constants._SCVI_VERSION_KEY]
        version_split = source_version.split('.')

        if int(version_split[0]) >= 1 and int(version_split[1]) >= 1:
            return

        # need this if <1.1 model is resaved with >=1.1 as new registry is
        # updated on setup
        manager.registry[_constants._SCVI_VERSION_KEY] = source_version
