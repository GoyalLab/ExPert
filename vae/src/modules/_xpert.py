from typing import Iterable

from src.modules._base import (
    Encoder, 
    GeneEmbeddingEncoder,
    SplitEncoder,
    DecoderSCVI, 
    EmbeddingAligner,
    ContextClassAligner, 
    HierarchicalAligner,
    Classifier,
    ClassEmbedding,
    Reranker
)
import src.utils.io as io
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS, LOSS_KEYS, PREDICTION_KEYS
import src.utils.embeddings as emb_utils
import src.utils.common as co
from src.utils.augmentations import BatchAugmentation

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
        n_shared: int | None = None,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        cls_text_dict: dict | None = None,
        ctrl_class_idx: int | None = None,
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
        use_augmentation: bool = False,
        use_decoded_augmentation: bool = False,
        shuffle_context: bool = True,
        link_latents: bool = False,
        use_ctx_cls: bool = True,
        use_adv_ctx_cls: bool = True,
        automatic_loss_scaling: bool = False,
        extra_encoder_kwargs: dict | None = {},
        extra_decoder_kwargs: dict | None = {},
        extra_aligner_kwargs: dict | None = {},
        extra_cls_kwargs: dict | None = {},
        extra_cls_emb_kwargs: dict | None = {},
        extra_aug_kwargs: dict | None = {},
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

        # Setup external embeddings
        self.ctx_emb = torch.nn.Embedding.from_pretrained(ctx_emb, freeze=True) if ctx_emb is not None else None
        self.use_adapter = use_adapter
        if use_adapter and not link_latents:
            extra_cls_emb_kwargs['n_output'] = n_latent
        self.pretrained_emb = cls_emb
        self.cls_emb = ClassEmbedding(pretrained_emb=cls_emb, class_texts=cls_text_dict, **extra_cls_emb_kwargs)
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
        else:
            self.n_ctx, self.n_ctx_dim = 0, 0
        self.n_cls = self.cls_emb.shape[0]
        self.n_cls_dim = self.cls_emb.shape[-1]
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
            'n_dim_context_emb': self.n_ctx_dim
        }
        if encoder_type == 'default':
            encoder_cls = Encoder
        elif encoder_type == 'split':
            encoder_cls = SplitEncoder
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
        n_input_decoder = self.n_shared + n_continuous_cov * decode_covariates
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
            # Reset output dimensions
            self.n_shared = self.aligner.n_output        
        
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

        # Setup an additional classifier on z, purely supervised
        self.classifier = Classifier(
            n_input=n_latent,
            n_labels=self.n_labels,
            **extra_cls_kwargs,
        )
        
        # Setup context classifier
        if use_ctx_cls:
            self.ctx_cls = Classifier(
                n_input=n_latent,
                n_labels=n_batch,
            )
        # Setup adversial context classifier
        if use_adv_ctx_cls:
            self.adv_ctx_cls = Classifier(
                n_input=n_latent,
                n_labels=n_batch,
            )
       
        # Check loss strategies
        self.set_align_ext_emb_strategies(align_ext_emb_loss_strategy)

        # Check parameter setup
        self._check_setup()

        # ----- Debug -----
        self._step = 0
        # Top-k reranker
        self.reranker = Reranker(n_input=self.n_shared)
        
    def _check_setup(self, error_on_fail: bool = False):
        # TODO: include all checks
        def e(msg):
            if error_on_fail:
                raise ValueError(msg)
            else:
                log.warning(msg)
        # If we use a split latent, context embeddings should not be decoded
        if self.encoder_type == 'split':
            if self.decode_ctx_emb:
                e(f'Registered split latent ("encoder_type": {self.encoder_type}) with context conditional decoding ("self.decode_ctx_emb": {self.decode_ctx_emb}).\nPlease make sure this is what you want.')
        
    def get_device(self):
        # Set own device
        if not hasattr(self, 'parameters'):
            return None
        _device_list = list({p.device for p in self.parameters()})
        if len(_device_list) > 0:
            return _device_list[0]
        return None
    
    def freeze_module(self, key: str, soft_lr: float | None = None, optim: torch.nn.Module | None = None) -> None:
        """Freeze a given module like encoder or decoder."""
        module = getattr(self, key)
        # Can't freeze what we don't have
        if module is None or not isinstance(module, torch.nn.Module):
            log.warning(f'{module} is not a valid module, skipped freeze.')
            return
        # Hard freeze the module
        if soft_lr is None or optim is None:
            # Set module to eval mode
            module.eval()
            # Disable gradient flow for all module parameters
            for param in module.parameters():
                param.requires_grad = False
            log.info(f'Successfully froze {key} parameters.')
            # Label the module as frozen
            module.frozen = True
        # Change module learning rates
        else:
            log.info(f'Successfully soft froze {key} parameters with lr: {soft_lr}.')
            for group in optim.param_groups:
                if any(p in group["params"] for p in module.parameters()):
                    group["lr"] = soft_lr
        
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
        inference_out: dict[str, torch.Tensor] = self.z_encoder(encoder_input, *categorical_input, g=g, ctx_label=batch_index, context_emb=ctx_emb)
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
                'q_m_l': inference_out.get('q_m_l'),
                'q_v_l': inference_out.get('q_v_l'),
                'zl': inference_out.get('zl'),
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
        if self.decode_covariates and self.ctx_emb is not None:
            # Get inflated context embeddings for batch (either projected or raw)
            if self.decode_context_projection:
                ctx_emb = ctx_proj[batch_index.squeeze(-1)]
            else:
                ctx_emb = self.ctx_emb(batch_index).reshape(batch_index.shape[0], -1)
        else:
            ctx_emb = None
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
        return_logits: bool = True,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            
        # Add z classifier output
        z_cls_logits = self.classifier(_z)
        inference_outputs['logits'] = z_cls_logits
        # Use regular alignment
        align_out: dict[str, torch.Tensor] = self.aligner(
            x=_z, cls_emb=cls_emb,
            return_logits=return_logits
        )

        # Apply re-ranker if it was used during training
        if self.reranker.active:
            z2c = align_out.get(MODULE_KEYS.Z_SHARED_KEY)
            c2z = align_out.get(MODULE_KEYS.CLS_PROJ_KEY)
            logits = align_out.get(MODULE_KEYS.CLS_LOGITS_KEY)
            # Add top-k prediction
            align_out[PREDICTION_KEYS.PREDICTION_KEY] = self.top_k_predict(
                z2c=z2c, c2z=c2z, logits=logits,
                K=self.reranker.K, soft=False
            )
        # Add alignment output to inference
        inference_outputs.update(align_out)
        return inference_outputs
    
    def predict(
        self,
        z_shared: torch.Tensor,
        cls2z: torch.Tensor,
        gene_temp: float = 0.1,
        module_temp: float | None = 0.3,
        pathway_temp: float | None = 0.7,
        return_logits: bool = True
    ):  
        # Use pathway predictions
        if self.use_hierarchical_labels and self.use_hierarchical_clip:
            predictions = self._hierarchical_predict(
                z_shared, gene_emb=cls2z, 
                gene_temperature=gene_temp,
                module_temperature=module_temp,
                pathway_temperature=pathway_temp
            )
            # Return gene prediction logits only
            predictions = predictions[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
        elif self.use_hierarchical_labels and io.non_zero(module_temp) and io.non_zero(pathway_temp):
            predictions = self._predict_hierarchical(
                z_shared, 
                cls2z, 
                gene_temp=gene_temp, module_temp=module_temp, pathway_temp=pathway_temp, 
                return_logits=return_logits
            )
        # Use gene clip only
        else:
            predictions = self._predict_gene_only(z_shared, cls2z, gene_temp=gene_temp, return_logits=return_logits)
        # Return predictions
        return predictions
    
    @torch.no_grad()
    def _hierarchical_predict(self, align_out: dict[str, torch.Tensor]):
        """
        Hierarchical inference prediction
        pathway -> module -> gene (hard top-down decoding).
        """
        # 1) Extract alignment logits
        logits_s = align_out["logits_s"]              # (B, P) pathway
        logits_m = align_out["logits_m"]              # (B, M) module
        logits_c = align_out[MODULE_KEYS.CLS_LOGITS_KEY]  # (B, C) class

        # 2) Predict pathway first (coarsest level)
        pred_pw = logits_s.argmax(dim=1)   # (B,)
        
        # If pathway is misc, predict on class logits only

        # 3) Mask modules that do NOT belong to the predicted pathway
        # module2pathway: (M,)
        module2pw = self.module2pw.to(logits_m.device)  # (M,)

        # Build a (B, M) boolean mask: True = valid module for that sample
        valid_mod_mask = (module2pw.unsqueeze(0) == pred_pw.unsqueeze(1))  # (B, M)

        # Mask out invalid modules with a large negative number
        masked_logits_m = logits_m.masked_fill(~valid_mod_mask, float("-inf"))

        # Predict module using masked logits
        pred_mod = masked_logits_m.argmax(dim=1)  # (B,)

        # 4) Mask classes that do NOT belong to the predicted module
        # cls2module: (C,)
        cls2mod = self.cls2module.to(logits_c.device)  # (C,)

        valid_cls_mask = (cls2mod.unsqueeze(0) == pred_mod.unsqueeze(1))  # (B, C)
        masked_logits_c = logits_c.masked_fill(~valid_cls_mask, float("-inf"))

        # Final class prediction
        pred_cls = masked_logits_c.argmax(dim=1)  # (B,)

        return {
            "pathway": pred_pw,
            "module": pred_mod,
            PREDICTION_KEYS.PREDICTION_KEY: pred_cls,
            PREDICTION_KEYS.SOFT_PREDICTION_KEY: masked_logits_c,
        }

    def _predict_hierarchical(
        self,
        z_shared: torch.Tensor | None = None,
        cls2z: torch.Tensor | None = None,
        gene_temp: float = 0.1,
        module_temp: float = 0.3,
        pathway_temp: float = 0.7,
        return_logits: bool = True
    ):
        # TODO: cache temperatures from model if None
        # TODO: make this more efficient?
        # Normalize
        N = cls2z.size(0)
        z_norm = F.normalize(z_shared, dim=-1)
        gene_embeddings_norm = F.normalize(cls2z, dim=-1)
        # Gene similarities
        gene_logits = z_norm @ gene_embeddings_norm.T / gene_temp # (B, n_genes)
        # Pathway probabilities
        module_logits = z_norm @ self._get_pathway_embeddings(cls2z).T / pathway_temp
        # Module probabilities
        pathway_logits = z_norm @ self._get_module_embeddings(cls2z).T / module_temp
        # Cache probabilities straight from logits
        pathway_probs = F.softmax(module_logits, dim=-1)  # (B, n_pathways)
        module_probs = F.softmax(pathway_logits, dim=-1)  # (B, n_modules)
    
        # Map each gene to its pathway/module probabilities
        gene_pathway_probs = pathway_probs[:, self.cls2pw[:N]]  # (B, n_genes)
        gene_module_probs = module_probs[:, self.cls2module[:N]]  # (B, n_genes)
        
        gene_weights = gene_pathway_probs * gene_module_probs  # (B, n_genes)
        # Final scores
        final_scores = gene_logits * gene_weights
        # Return logits
        if return_logits:
            return final_scores
        else:
            # Return predictions
            return final_scores.argmax(dim=-1)
        
    def _predict_gene_only(
        self,
        z_shared: torch.Tensor,
        cls2z: torch.Tensor,
        gene_temp: float = 0.1,
        return_logits: bool = True
    ):
        # Get model's current gene embeddings
        gene_embeddings = cls2z
        # Normalize
        z_norm = F.normalize(z_shared, dim=-1)
        gene_embeddings_norm = F.normalize(gene_embeddings, dim=-1)
        # Gene similarities
        gene_sims = z_norm @ gene_embeddings_norm.T / gene_temp # (B, n_genes)
        # Return logits
        if return_logits:
            return gene_sims
        else:
            # Return predictions
            return gene_sims.argmax(dim=-1)
    
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

    def _random_unseen_replacement(
        self,
        ctx_idx: torch.Tensor,
        cls_idx: torch.Tensor,
        p: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly replace some labels with unseen labels to keep zero-shot active.""" 
        if self.has_unseen_ctx:
            ctx_idx = emb_utils.replace_with_unseen_labels(ctx_idx, n_seen=self.n_batch, n_total=self.n_ctx, p=p)
        if self.has_unseen_cls:
            cls_idx = emb_utils.replace_with_unseen_labels(cls_idx, n_seen=self.n_labels, n_total=self.n_cls, p=p)
        return ctx_idx, cls_idx
    
    def _observable_bias_loss(self, random_logits: torch.Tensor, n_obs: int):
        """Punish random predictions that are biased towards observed classes."""
        # Get number of all available classes
        # Calculate probabilities
        probs = random_logits.softmax(dim=-1)
        preds = random_logits.argmax(dim=-1)
        # Get mean density in observed fraction
        frac_prob_obs = probs[:, :n_obs].sum(dim=-1).mean()
        frac_pred_obs = (preds < n_obs).float().mean()
        # Mean expected density
        expected_frac = n_obs / random_logits.size(-1)
        # Return bias loss
        soft_loss = (frac_prob_obs - expected_frac)**2
        hard_loss = (frac_pred_obs - expected_frac)**2
        return soft_loss + hard_loss
    
    def _logit_variance_loss(self, random_logits: torch.Tensor, m: float = 1e-2):
        """Prevent model from pushing down values."""
        var = random_logits.var(dim=-1)
        # penalize too-flat logits
        return torch.relu(m - var).mean()
    
    def get_ctx_buffer(self, reset: bool = False):
        """Return and optionally clear the stored pseudo argmax buffer for context predictions."""
        buffer_name = f"{self._ctx_key}_buffer"
        if not hasattr(self, buffer_name):
            return None

        buf = getattr(self, buffer_name)

        if reset:
            # reset the existing buffer without re-registering
            empty = torch.empty(0, dtype=buf.dtype, device=buf.device)
            setattr(self, buffer_name, empty)

        return buf

    def get_cls_buffer(self, reset: bool = False):
        """Return and optionally clear the stored pseudo argmax buffer for class predictions."""
        buffer_name = f"{self._cls_key}_buffer"
        if not hasattr(self, buffer_name):
            return None

        buf = getattr(self, buffer_name)

        if reset:
            empty = torch.empty(0, dtype=buf.dtype, device=buf.device)
            setattr(self, buffer_name, empty)

        return buf

    def _update_pseudo_buffer(self, random_logits: torch.Tensor, name: str, T: float = 0.1):
        """Append new argmax predictions to a named buffer."""
        preds = random_logits.argmax(dim=-1).detach()
        buf_name = f"{name}_buffer"

        # create or update the buffer safely
        if not hasattr(self, buf_name):
            # register once (persistent, moves with .to(device))
            self.register_buffer(buf_name, preds.clone())
        else:
            buf = getattr(self, buf_name)
            preds = preds.to(buf.device)
            new_buf = torch.cat((buf, preds), dim=0)
            setattr(self, buf_name, new_buf)

    def _grad_reversed_entropy_loss(self, logits: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Maximize entropy via gradient reversal."""
        logits_rev = co.grad_reverse(logits, lam)
        p = torch.softmax(logits_rev, dim=-1)
        return (p * p.log()).sum(dim=-1).mean()
    
    def context_loss(
            self,
            z: torch.Tensor,
            context_labels: torch.Tensor,
            reduction: str | None = 'mean'
        ) -> torch.Tensor:
        """
        Context classification loss: encourage encoder to include context information.
        """
        if not getattr(self, 'ctx_cls', False):
            return torch.tensor(1e-5, device=z.device)

        # Predict context
        context_logits: torch.Tensor = self.ctx_cls(z)

        # Set labels
        if len(context_labels.shape) > 1:
            # Reshape labels to 1d array
            context_labels = context_labels.reshape(-1)
            
        # Get predictions
        predictions = context_logits.argmax(dim=-1)

        # Normal cross-entropy loss for the context head
        reduction = reduction if reduction else self.non_elbo_reduction
        loss = F.cross_entropy(context_logits, context_labels, reduction='none')
        return self._reduce_loss(loss, reduction), predictions.flatten(), context_labels.flatten()
    
    def adversarial_context_loss(
            self,
            z: torch.Tensor,
            context_labels: torch.Tensor,
            lambda_adv: float = 0.1,
        ) -> torch.Tensor:
        """
        Adversarial context loss: encourage encoder to remove context information.
        The gradient is reversed before passing through the context classifier.
        """
        if not getattr(self, 'adv_ctx_cls', False):
            return torch.tensor(1e-5, device=z.device)

        # Reverse gradient before classification
        z_rev = co.grad_reverse(z, lambda_adv)

        # Predict context
        context_logits = self.adv_ctx_cls(z_rev)

        # Set labels
        if len(context_labels.shape) > 1:
            # Reshape labels to 1d array
            context_labels = context_labels.reshape(-1)
            
        # Get predictions
        predictions = context_logits.argmax(dim=-1)

        # Normal cross-entropy loss for the context head
        loss = F.cross_entropy(context_logits, context_labels, reduction='none')
        return self._reduce_loss(loss, self.non_elbo_reduction), predictions.flatten(), context_labels.flatten()

    def _bias_bce(self, pseudo_logits: torch.Tensor, n_obs: int, n_cls: int) -> torch.Tensor:
        """
        bias loss: penalize if random pseudo alignments
        can distinguish observed vs unobserved classes.
        """
        device = pseudo_logits.device
        B, C = pseudo_logits.shape
        assert C == n_cls

        # true indicator: which classes are observed (0/1 mask, fixed)
        obs_mask = torch.zeros((C,), device=device)
        obs_mask[:n_obs] = 1.0  # 1 for observed, 0 for unseen

        # soft prediction for "observed-ness"
        # this measures how much mass the model assigns to observed classes
        probs = torch.sigmoid(pseudo_logits)
        p_obs = (probs * obs_mask).sum(-1, keepdim=True) / (obs_mask.sum() + 1e-6)

        # target: 0.5 for all --> confusion objective
        target = torch.full_like(p_obs, 0.5)
        # model wants p_obs around 0.5 (no bias)
        return F.mse_loss(p_obs, target)

    def _ce_cls_loss(self, logits: torch.Tensor, y: torch.Tensor, T: float | None = None) -> torch.Tensor:
        """Calculate additional simple classification loss on initial latent space."""
        # Get classifier temperature
        if T is None:
            T = self.classifier.temperature
        # Return an CE loss
        loss = F.cross_entropy(logits, y, reduction='none')
        return loss
    
    def _center_z_on_ctrl(self, z: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Center z on mean of control cells."""
        if self.ctrl_class_idx is None:
            return z, y, b
        # Get control cell mask
        ctrl_mask = (y == self.ctrl_class_idx).squeeze(-1)
        # Center z on control cell mean TODO: optionally do this per context
        zs = []
        for ctx in torch.unique(b):
            mask_ctx = (b == ctx).squeeze(-1)
            mask_ctrl = ctrl_mask & mask_ctx
            mask_no_ctrl = ~ctrl_mask & mask_ctx
            if mask_ctrl.any():
                z_centered = z[mask_no_ctrl] - z[mask_ctrl].mean(0, keepdim=True)
                zs.append(z_centered)
        zs = torch.cat(zs, dim=0)
        y = y[~ctrl_mask]
        b = b[~ctrl_mask]
        return zs, y, b
    
    def _split_batch(self, x: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split batch into class and control."""
        if self.ctrl_class_idx is None:
            return x, None, y, b
        # Get control cell mask
        ctrl_mask = (y == self.ctrl_class_idx).squeeze(-1)
        if not ctrl_mask.any():
            return x, None, y, b
        # Split batch
        return {
            'ctrl': {
                MODULE_KEYS.X_KEY: x[ctrl_mask],
                MODULE_KEYS.LABEL_KEY: y[ctrl_mask],
                REGISTRY_KEYS.BATCH_KEY: b[ctrl_mask]
            },
            'class': {
                MODULE_KEYS.X_KEY: x[~ctrl_mask],
                MODULE_KEYS.LABEL_KEY: y[~ctrl_mask],
                REGISTRY_KEYS.BATCH_KEY: b[~ctrl_mask]
            }
        }

    def _contr_ctrl_loss(self, z: torch.Tensor, y: torch.Tensor, b: torch.Tensor, T: torch.Tensor = 1.0) -> torch.Tensor:
        """Apply a contrastive loss from z to control z."""
        if self.ctrl_class_idx is None:
            return z, y, b, torch.tensor(0.0)
        # Calculate control mask
        ctrl_mask = (y == self.ctrl_class_idx).squeeze(-1)
        if not ctrl_mask.any():
            return z, y, b, torch.tensor(0.0)
        # Split batch labels into z and z ctrl
        _y = y[~ctrl_mask]
        _b = b[~ctrl_mask]
        # Normalize embeddings
        z = F.normalize(z, dim=-1)

        # Masks
        ctrl_mask = (y == self.ctrl_class_idx).flatten()
        if not ctrl_mask.any() or (~ctrl_mask).sum() < 2:
            return torch.tensor(0.0, device=z.device)
        # Split latent into z and control z
        z_ctrl, z_noctrl = z[ctrl_mask], z[~ctrl_mask]
        ctx_ctrl, ctx_noctrl = b[ctrl_mask], b[~ctrl_mask]

        # global positive similarities among non-controls
        sim_pos = z_noctrl @ z_noctrl.T / T
        mask_self = torch.eye(sim_pos.size(0), device=z.device)
        # Class mask
        same_class = (_y == _y.T).float()
        pos_mask = same_class * (1 - mask_self)

        # local negatives = control samples from same context only
        sim_neg = z_noctrl @ z_ctrl.T / T
        same_ctx = (ctx_ctrl.flatten() == ctx_noctrl).float()

        # InfoNCE: positives (other perturbations) vs context-matched controls
        pos = torch.exp(sim_pos).sum(dim=-1)
        neg = torch.exp(sim_neg * same_ctx).sum(dim=-1)
        log_probs = sim_pos - torch.log(neg + 1e-8)
        loss_ps = -(log_probs * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-08)
        loss = self._reduce_loss(loss_ps, reduction=self.non_elbo_reduction)
        return z_noctrl, _y, _b, loss
    
    def _clip_loss(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        emb: torch.Tensor,
        T: float = 0.1,
        k: int = 30,
        unique_proxies: bool = True,
        reduce: bool = True,
        return_logits: bool = True,
        use_reverse: bool = True,
        n_unseen: int = 10,
        unseen_weight: float = 0.2,
        unseen_indices: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        use_random_proxies: bool = False,
    ):
        """
        U-CLIP loss with unique-class prototypes per batch + hard negative mining.
        
        Args:
            z: (B, d) - cell embeddings from scRNA encoder
            y: (B,) - class labels (which gene was perturbed)
            emb: (C_all, d) - FULL class embedding matrix (ALL gene descriptions, seen + unseen)
            T: temperature parameter
            k: number of hard negatives to mine (None or 0 for full softmax)
            unique_proxies: if True, only use unique classes in batch (U-CLIP)
            return_full: if True, return combined loss; if False, return dict
            use_reverse: if True, compute symmetric class->cell loss
            seen_indices: (C_seen,) - indices of seen/training genes in emb (optional)
                         If provided, uses unseen genes as hard negatives
            n_unseen: number of unseen negatives to sample per batch
            unseen_weight: weight for unseen negative loss (0.0-1.0)
        
        Returns:
            loss: scalar if return_full=True
            (loss_z2c, loss_c2z): tuple if return_full=False
        """
        device = z.device
        y = y.flatten()
        B = z.size(0)
        
        # Normalize latent and class embeddings
        z = F.normalize(z, dim=-1)
        emb_norm = F.normalize(emb, dim=-1)  # Normalize full embedding matrix
        
        # Split seen and unseen indices
        use_unseen_negatives = unseen_indices is not None and unseen_weight > 0
        
        # Unique-class CLIP (U-CLIP)
        if unique_proxies:
            unique_classes, inv = torch.unique(y, return_inverse=True)
            # U = number of unique classes in the batch
            proxies = emb_norm[unique_classes]  # (U, d)
            # positive target index is inv (mapping sample -> unique prototype)
            targets = inv
        else:
            # Use per-sample embeddings (no uniqueness)
            proxies = emb_norm[y]  # (B, d)
            targets = torch.arange(B, device=device)
        # Randomize proxies
        if use_random_proxies:
            C = emb.size(0)
            M = proxies.size(0)
            proxies = emb_norm[torch.randperm(C)[:M]]
        
        # Normalize proxies
        proxies = F.normalize(proxies, dim=-1)
        U = proxies.size(0)
        
        # In case of multi-proxies, do logsumexp
        if proxies.ndim == 3:
            sim = torch.einsum("bd,cmd->bcm", z, proxies)   # (B, C, M)
            logits_z2c = torch.logsumexp(sim / T, dim=-1)  # (B, C)
        else:
            # Forward direction: z -> class (cell -> gene description)
            logits_z2c = (z @ proxies.T) / T  # (B, U)
            
        # Log margin between pos and neg
        pos_sim = logits_z2c[torch.arange(B), targets].mean()
        neg_sim = logits_z2c.mean()
        margin = pos_sim / (neg_sim + 1e-5)
        
        # Hard negative mining
        if k is not None and k > 0:
            # -------------------------------------------
            # Hard Negative Mining over prototypes
            # -------------------------------------------
            # For each sample, exclude the positive prototype
            mask = torch.zeros_like(logits_z2c, dtype=torch.bool)
            mask[torch.arange(B), targets] = True
            
            # Mask out positive class by setting to very low value
            neg_scores = logits_z2c.masked_fill(mask, -1e9)
            
            # Select top-k hardest negatives
            k_eff = int(min(k, U - 1))
            hard_negs = neg_scores.topk(k_eff, dim=-1).values  # (B, k)
            
            # Get positive scores
            pos = logits_z2c[torch.arange(B), targets].unsqueeze(1)  # (B, 1)
            
            # Concatenate: [positive, hard_neg_1, ..., hard_neg_k]
            logits_z2c_mined = torch.cat([pos, hard_negs], dim=1)  # (B, k+1)
            
            # Positive class is always index 0
            loss_z2c = F.cross_entropy(
                logits_z2c_mined.clamp(-20, 20),
                torch.zeros(B, dtype=torch.long, device=device),
                reduction='none',
                label_smoothing=label_smoothing
            )
        else:
            # Standard full-softmax over all U classes
            loss_z2c = F.cross_entropy(logits_z2c.clamp(-20, 20), targets, reduction='none', label_smoothing=label_smoothing)
        
        # UNSEEN NEGATIVES: Add unseen gene embeddings as hard negatives
        loss_z2c_unseen = None
        if use_unseen_negatives and len(unseen_indices) > 0 and n_unseen > 0:
            # Sample random unseen embeddings
            n_unseen_eff = int(min(n_unseen, len(unseen_indices)))
            # Move indices to current device
            unseen_indices = unseen_indices.to(device)
            # Randomly sample unseen indices to use
            sampled_unseen_indices = unseen_indices[torch.randperm(len(unseen_indices), device=device)[:n_unseen_eff]]
            unseen_sample = emb_norm[sampled_unseen_indices]  # (n_unseen, d)
            
            # Augmented class embeddings: [seen_proxies, unseen_proxies]
            augmented_proxies = torch.cat([proxies, unseen_sample], dim=0)  # (U + n_unseen, d)
            # In case of multi-proxies, do logsumexp instead of basic dot product
            if augmented_proxies.ndim == 3:
                sim_aug = torch.einsum("bd,cmd->bcm", z, augmented_proxies)   # (B, C, M)
                logits_z2c_aug = torch.logsumexp(sim_aug / T, dim=-1)  # (B, C)
            else:
                # Compute augmented logits
                logits_z2c_aug = (z @ augmented_proxies.T) / T  # (B, U + n_unseen)
            
            if k is not None and k > 0:
                # Hard negative mining with unseen
                mask_aug = torch.zeros_like(logits_z2c_aug, dtype=torch.bool)
                mask_aug[torch.arange(B), targets] = True
                
                neg_scores_aug = logits_z2c_aug.masked_fill(mask_aug, -1e9)
                
                # Select top-k from all negatives (seen + unseen)
                k_eff_aug = int(min(k, U + n_unseen_eff - 1))
                hard_negs_aug = neg_scores_aug.topk(k_eff_aug, dim=-1).values  # (B, k)
                
                # Positive scores (same as before)
                pos_aug = logits_z2c_aug[torch.arange(B), targets].unsqueeze(1)  # (B, 1)
                
                logits_z2c_aug_mined = torch.cat([pos_aug, hard_negs_aug], dim=1)
                
                loss_z2c_unseen = F.cross_entropy(
                    logits_z2c_aug_mined.clamp(-20, 20),
                    torch.zeros(B, dtype=torch.long, device=device),
                    reduction='none',
                    label_smoothing=label_smoothing
                )
            else:
                # Full softmax with augmented classes
                loss_z2c_unseen = F.cross_entropy(
                    logits_z2c_aug.clamp(-20, 20), 
                    targets, 
                    reduction='none',
                    label_smoothing=label_smoothing
                )
        
        # Reverse direction: class -> z (gene description -> cell)
        logits_c2z = None
        if use_reverse:
            if unique_proxies:
                # In case of multi-proxies, do logsumexp
                if proxies.ndim == 3:
                    sim_aug_rev = torch.einsum("cmd,bd->cbm", proxies, z)   # (C, B, M)
                    logits_c2z = torch.logsumexp(sim_aug_rev / T, dim=-1)  # (C, B)
                else:
                    # Proxies -> cells: (U, d) @ (d, B) = (U, B)
                    logits_c2z = (proxies @ z.T) / T  # (U, B)
                
                # For each unique class, compute loss over its corresponding cells
                loss_c2z_list = []
                
                for u_idx in range(U):
                    # Find which samples belong to this unique class
                    sample_mask = (inv == u_idx)  # (B,) boolean mask
                    num_pos = sample_mask.sum().item()
                    
                    if num_pos == 0:
                        continue
                    
                    class_logits = logits_c2z[u_idx]  # (B,) - similarity to all cells
                    
                    if k is not None and k > 0:
                        # Hard negative mining for reverse direction
                        # Positives: cells with this class label
                        # Negatives: all other cells
                        
                        pos_scores = class_logits[sample_mask]  # (num_pos,)
                        neg_scores = class_logits[~sample_mask]  # (B - num_pos,)
                        
                        if len(neg_scores) > 0:
                            # Select top-k hardest negatives
                            k_eff = int(min(k, len(neg_scores)))
                            hard_negs = neg_scores.topk(k_eff).values  # (k,)
                            
                            # For each positive cell, contrast with hard negatives
                            # Shape: (num_pos, 1+k)
                            pos_expanded = pos_scores.unsqueeze(1)  # (num_pos, 1)
                            hard_negs_expanded = hard_negs.unsqueeze(0).expand(num_pos, -1)  # (num_pos, k)
                            
                            logits_mined = torch.cat([pos_expanded, hard_negs_expanded], dim=1)
                            
                            # Each positive should match index 0
                            loss = F.cross_entropy(
                                logits_mined.clamp(-20, 20),
                                torch.zeros(num_pos, dtype=torch.long, device=device),
                                reduction='none',
                                label_smoothing=label_smoothing
                            )
                            loss_c2z_list.append(loss)
                    else:
                        # Standard softmax over all cells
                        # Target: uniform distribution over positive samples
                        targets_soft = sample_mask.float() / num_pos  # (B,)
                        log_probs = F.log_softmax(class_logits, dim=0)  # (B,)
                        loss = -(targets_soft * log_probs).sum()
                        loss_c2z_list.append(loss.reshape(1))
                
                # Average loss across unique classes
                if loss_c2z_list:
                    loss_c2z = torch.cat(loss_c2z_list)
                else:
                    loss_c2z = torch.zeros(1, device=device, requires_grad=True)
            
            else:
                # Per-sample mode (not using unique proxies)
                # Each sample embedding should match its corresponding cell
                labels = torch.arange(B, device=device)
                chosen = emb_norm[y]  # (B, d)
                logits_c2z = (chosen @ z.T) / T  # (B, B)
                loss_c2z = F.cross_entropy(logits_c2z.clamp(-20, 20), labels, reduction='none', label_smoothing=label_smoothing)
        else:
            # No reverse loss
            loss_c2z = torch.zeros(1, device=device, requires_grad=True)
        
        # ---------------------------------------------------------
        # Combine losses
        # ---------------------------------------------------------
        o = {'clip_margin': margin}
        # Add logits to output
        if return_logits:
            o.update({'logits': {
                LOSS_KEYS.Z2C_LOGITS: logits_z2c,
                LOSS_KEYS.C2Z_LOGITS: logits_c2z
            }})
        # Add reduced loss to output
        if reduce:
            # Combine forward losses (seen + unseen)
            if loss_z2c_unseen is not None:
                loss_z2c_combined = (
                    (1 - unseen_weight) * self._reduce_loss(loss_z2c, self.non_elbo_reduction) +
                    unseen_weight * self._reduce_loss(loss_z2c_unseen, self.non_elbo_reduction)
                )
            else:
                loss_z2c_combined = self._reduce_loss(loss_z2c, self.non_elbo_reduction)
            
            if use_reverse:
                # Symmetric: average both directions
                o[LOSS_KEYS.LOSS] = 0.5 * (
                    loss_z2c_combined + 
                    self._reduce_loss(loss_c2z, self.non_elbo_reduction)
                )
            else:
                # Asymmetric: only forward direction
                o[LOSS_KEYS.LOSS] = loss_z2c_combined
        else:
            # Return all components (useful for logging/caching)
            o.update({
                LOSS_KEYS.Z2C: loss_z2c,
                LOSS_KEYS.C2Z: loss_c2z,
                LOSS_KEYS.UNSEEN_Z2C: loss_z2c_unseen,  
            })
        # Return full output
        return o
    
    def hierarchical_consistency_loss(
        self,
        logits_cls: torch.Tensor,
        logits_m: torch.Tensor,
        logits_p: torch.Tensor,
        pw_y: torch.Tensor | None = None,
        mod_y: torch.Tensor | None = None,
        eps: float = 1e-8,
        detach_targets: bool = True,
    ):
        # probs
        p_c  = F.softmax(logits_cls, dim=-1)          # (B, C)
        p_m  = F.softmax(logits_m, dim=-1).clamp_min(eps)   # (B, M)
        p_pw = F.softmax(logits_p, dim=-1).clamp_min(eps)   # (B, P)

        cls2mod = self.cls2module.to(logits_cls.device)   # (C,)
        cls2pw  = self.cls2pw.to(logits_cls.device)       # (C,)
        module2pw = self.module2pw.to(logits_cls.device)  # (M,)

        B, C = p_c.shape
        n_mod = int(cls2mod.max().item()) + 1
        n_pw  = int(cls2pw.max().item())  + 1

        # classes -> modules
        p_m_from_c = torch.zeros(B, n_mod, device=p_c.device)
        p_m_from_c.index_add_(1, cls2mod, p_c)
        p_m_from_c = p_m_from_c / p_m_from_c.sum(dim=-1, keepdim=True).clamp_min(eps)

        # modules -> pathways (preferred)
        p_pw_from_m = torch.zeros(B, n_pw, device=p_c.device)
        p_pw_from_m.index_add_(1, module2pw, p_m_from_c)
        p_pw_from_m = p_pw_from_m / p_pw_from_m.sum(dim=-1, keepdim=True).clamp_min(eps)

        # targets (optionally detached)
        p_m_tgt  = p_m_from_c.detach() if detach_targets else p_m_from_c
        p_pw_tgt = p_pw_from_m.detach() if detach_targets else p_pw_from_m

        # KL(p_direct || p_target)
        L_cons_mod = F.kl_div(p_m.log(),  p_m_tgt.clamp_min(eps),  reduction="none").sum(dim=-1)
        L_cons_pw  = F.kl_div(p_pw.log(), p_pw_tgt.clamp_min(eps), reduction="none").sum(dim=-1)

        # Optional: mask misc samples
        if pw_y is not None and getattr(self, "misc_pw_idx", -1) > -1:
            keep = (pw_y != self.misc_pw_idx).float()
            L_cons_pw = L_cons_pw * keep
        if mod_y is not None and getattr(self, "misc_mod_idx", -1) > -1:
            keep = (mod_y != self.misc_mod_idx).float()
            L_cons_mod = L_cons_mod * keep
            
        # Combine losses
        L_cons = L_cons_mod + L_cons_pw

        return L_cons
    
    def hierarchical_clip_loss(
        self,
        align_out: dict[str, torch.Tensor],
        labels: torch.Tensor,
        w_pw: float = 1.0,
        w_mod: float = 1.0,
        w_cls: float = 1.0,
        w_cons: float = 0.1,
        label_smoothing: float | None = 0.1,
        reduce: bool = True,
    ):
        # Get hierachical labels
        y = labels.flatten()
        pw_y = self.cls2pw[y]
        mod_y = self.cls2module[y]
        
        # Extract logits from alignment
        logits_cls = align_out[MODULE_KEYS.CLS_LOGITS_KEY]
        logits_mod = align_out['logits_m']
        logits_pw = align_out['logits_s']
        # Calculate forward clip losses
        L_pw = F.cross_entropy(logits_pw, pw_y, reduction='none', label_smoothing=label_smoothing)
        L_mod = F.cross_entropy(logits_mod, mod_y, reduction='none', label_smoothing=label_smoothing)
        L_cls = F.cross_entropy(logits_cls, y, reduction='none')
        
        # Set misc modules and pathway losses to 0
        if self.misc_pw_idx > -1:
            L_pw = L_pw * (pw_y != self.misc_pw_idx).float()
        if self.misc_mod_idx > -1:
            L_mod = L_mod * (mod_y != self.misc_mod_idx).float()
            
        # Calculate consistency losses
        if w_cons > 0:
            L_cons = self.hierarchical_consistency_loss(logits_cls, logits_mod, logits_pw, pw_y=pw_y, mod_y=mod_y)
        else:
            L_cons = torch.tensor(0.0)
        
        # Reduce losses
        if reduce:
            L_pw = self._reduce_loss(L_pw, self.non_elbo_reduction)
            L_mod = self._reduce_loss(L_mod, self.non_elbo_reduction)
            L_cls = self._reduce_loss(L_cls, self.non_elbo_reduction)
            L_cons = self._reduce_loss(L_cons, self.non_elbo_reduction)
            
        # Combine losses
        L = w_pw * L_pw + w_mod * L_mod + w_cls * L_cls + w_cons * L_cons
        
        # Create output dictionary
        return {
            LOSS_KEYS.LOSS: L,
            'L_pw': L_pw, 'L_mod': L_mod, 'L_cls': L_cls,
            'L_cons': L_cons
        }

    def laplacian_smoothness_loss(
        self,
        P: torch.Tensor,          # (C, d) class prototypes
    ):
        """
        Regularize class prototypes to follow pre-computed similarities
        """
        if not hasattr(self, 'laplacian_graph'):
            return torch.tensor(0.0)
        L = self.laplacian_graph
        L_graph = torch.trace(P.T @ L @ P)
        return self._reduce_loss(L_graph, self.non_elbo_reduction)
    
    def top_k_predict(
        self,
        z2c: torch.Tensor,
        c2z: torch.Tensor,
        logits: torch.Tensor,
        K: int = 10,
        soft: bool = False
    ):  
        B = logits.shape[0]
        # Re-rank top-k predictions based on concatenated inputs
        _, idx = torch.topk(logits, K, dim=-1)
        # Get selected batch
        z_sel = z2c.unsqueeze(1).expand(-1, K, -1)       # (B, K, d)
        t_sel = c2z[idx]                                 # (B, K, d)
        s_sel = logits.gather(1, idx).unsqueeze(-1)      # (B, K, 1)
        # Get re-ranked logits from small network
        logits_rerank = self.reranker(z_sel, t_sel, s_sel)   # (B, K)
        predictions = idx[torch.arange(B), logits_rerank.argmax(dim=-1)]
        if soft:
            return {
                PREDICTION_KEYS.PREDICTION_KEY: predictions,
                PREDICTION_KEYS.SOFT_PREDICTION_KEY: logits_rerank,
            }
        else:
            return predictions
    
    def re_rank_loss(
        self,
        z2c: torch.Tensor,      # (B, d)  RNA embedding used for retrieval
        c2z: torch.Tensor,      # (C, d)  class embeddings/prototypes
        logits: torch.Tensor,   # (B, C)  retrieval logits (e.g., CLIP logits)
        y: torch.Tensor,        # (B,)    true class indices
        K: int = 10,
        margin: float = 0.1,
        w_inclusion: float = 0.5,
    ):
        """
        Teacher-forced reranking:
        - Candidate set always contains the positive at position 0.
        - Remaining K-1 are hard negatives from retrieval (topK excluding y).
        Also adds a differentiable top-K inclusion penalty that pushes the true
        class score above the K-th score.

        Returns:
        LOSS: reduced rerank + inclusion penalty
        PREDICTION_KEY: predicted class id (from reranked candidates)
        (optionally diagnostics)
        """
        B, C = logits.shape
        device = logits.device

        # ------------------------------------------------------------
        # 1) Differentiable top-K inclusion penalty on retrieval logits
        #    Encourage s_pos >= s_k + margin
        # ------------------------------------------------------------
        # K-th threshold among *retrieval* logits
        topk_vals, idx_top = torch.topk(logits, K, dim=-1)                 # (B, K)
        s_k = topk_vals[:, -1]                                       # (B,)
        s_pos = logits.gather(1, y.unsqueeze(1)).squeeze(1)          # (B,)
        L_inclusion_ps = F.softplus((s_k - s_pos) + margin)          # (B,)
        
        # ------------------------------------------------------------
        # 2) Teacher-forced candidate set: [y] + (K-1) hard negatives
        # ------------------------------------------------------------
        # Remove y from idx_top if present
        keep = idx_top != y.unsqueeze(1)                             # (B, K) bool
        # Replace filtered-out positions with sentinel C (out of range) then sort to push them to the end
        idx_tmp = idx_top.clone()
        idx_tmp[~keep] = C                                           # sentinel

        # Stable-ish way to take first K-1 after filtering:
        # Sort by (is_sentinel, original_rank) using sentinel value C (largest) as key
        idx_sorted, _ = torch.sort(idx_tmp, dim=1)                   # (B, K) sentinel moves to end
        idx_neg = idx_sorted[:, : max(K - 1, 0)]                     # (B, K-1)

        # If y was not in the topK, we still have K-1 valid negatives already.
        # If y WAS in topK, sentinel removed it and we still have K-1 negatives unless duplicates reduce it.
        # Edge case: if C is tiny or logits has duplicates causing too few unique negatives, fall back to random fill.
        if K > 1:
            need_fill = (idx_neg == C).any(dim=1)  # rows where we didn't get enough negatives
            if need_fill.any():
                # random negatives excluding y (cheap fallback)
                # NOTE: if C is huge, this is fine; if C ~ K, you may still get repeats.
                rand = torch.randint(0, C, (need_fill.sum().item(), K - 1), device=device)
                # avoid y
                y_sub = y[need_fill].unsqueeze(1)
                rand = torch.where(rand == y_sub, (rand + 1) % C, rand)
                idx_neg[need_fill] = torch.where(idx_neg[need_fill] == C, rand, idx_neg[need_fill])

        # Prepend positive explicitly at position 0
        idx = torch.cat([y.unsqueeze(1), idx_neg], dim=1)            # (B, K)
        target_idx = torch.zeros(B, dtype=torch.long, device=device) # positive is always position 0

        # ------------------------------------------------------------
        # 3) Build reranker features and compute rerank CE
        # ------------------------------------------------------------
        z_sel = z2c.unsqueeze(1).expand(-1, K, -1)                   # (B, K, d)
        t_sel = c2z[idx]                                             # (B, K, d)
        s_sel = logits.gather(1, idx).unsqueeze(-1)                  # (B, K, 1)

        logits_rerank = self.reranker(z_sel, t_sel, s_sel)           # (B, K)

        L_rerank_ps = F.cross_entropy(logits_rerank, target_idx, reduction="none")  # (B,)

        # ------------------------------------------------------------
        # 4) Combine + reduce
        # ------------------------------------------------------------
        # Per-sample combined
        L_total_ps = L_rerank_ps + (w_inclusion * L_inclusion_ps)

        out = {
            LOSS_KEYS.LOSS: self._reduce_loss(L_total_ps, self.non_elbo_reduction),
            "L_rerank": self._reduce_loss(L_rerank_ps, self.non_elbo_reduction),
            "L_inclusion": self._reduce_loss(L_inclusion_ps, self.non_elbo_reduction),
            # Diagnostics (optional but helpful)
            "topk_inclusion_rate": (s_pos >= s_k).float().mean().detach(),
        }
        return out
    
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
        x: torch.Tensor = augmented_inputs[MODULE_KEYS.X_KEY]
        # Get model inference
        if self.encoder_type == 'split':
            if self.log_variational:
                x = torch.log1p(x)
            inference_outputs = self.z_encoder.local_encoder(x, g=g)
        else:
            inference_outputs = self.inference(**augmented_inputs, g=g)
        # Align augmented batch
        zk, qzk = MODULE_KEYS.Z_KEY, MODULE_KEYS.QZ_KEY
        x: torch.Tensor = inference_outputs[qzk].loc if use_posterior_mean else inference_outputs[zk]
        aug_align_out: dict[str, torch.Tensor] = self.aligner(x, cls_emb, T=T, return_logits=False)
        aug_align_out.update(augmented_inputs)
        return aug_align_out
    
    def augmentation_consistency_loss(
        self, 
        inference_outputs: dict[str, torch.Tensor],
        alignment_outputs: dict[str, torch.Tensor] | None = None,
        use_posterior_mean: bool = True,
    ):
        # Get augmented batch with artifical noise / dropout
        if self.augmentation is not None:
            aug_batch = self.augmentation(inference_outputs)
            n_aug = self.augmentation.n_augmentations
        # Get augmented batch from decoder
        elif self.use_decoded_augmentation:
            aug_batch = self._get_decoded_augmentation(inference_outputs)
            n_aug = 1
        else:
            return torch.tensor(0.0)
        # Do second forward pass with augmented batch
        aug_out = self._regular_inference(**aug_batch)
        # Choose local latent if possible
        qz_key = 'qzl' if 'qzl' in inference_outputs else MODULE_KEYS.QZ_KEY
        z_key = 'zl' if 'zl' in inference_outputs else MODULE_KEYS.Z_KEY
        # Get inference output for real and augmented batch
        if use_posterior_mean:
            real = inference_outputs[qz_key].loc
            aug = aug_out[qz_key].loc
        else:
            real = inference_outputs[z_key]
            aug = aug_out[z_key]
        # Compare clip spaces if available
        if alignment_outputs is not None:
            real = alignment_outputs[MODULE_KEYS.CLS_LOGITS_KEY]
            aug = self.aligner(aug, cls_emb=self.cached_cls_emb, return_logits=True)[MODULE_KEYS.CLS_LOGITS_KEY]
        # Expand mu according to number of augmentations
        if n_aug > 1:
            real = real.repeat_interleave(n_aug, dim=0)  # (B * n_aug, D)
        # Detach real signal
        real = real.detach()
        if alignment_outputs is not None:
            p_real = F.softmax(real, dim=-1).detach()
            p_aug  = F.log_softmax(aug, dim=-1)
            # Get KL loss
            L_cons_ps = F.kl_div(p_aug, p_real, reduction="none")
        else:
            # Normalize latents
            real = F.normalize(real, dim=-1)
            aug = F.normalize(aug, dim=-1)
            # Get loss
            L_cons_ps = 1.0 - (real * aug).sum(-1)
        return self._reduce_loss(L_cons_ps, self.non_elbo_reduction)

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
    
    def _center_z(self, z: torch.Tensor, ctx: torch.Tensor, m: float = 0.95) -> torch.Tensor:
        """
        Center z by subtracting a running (EMA) context-specific mean.

        Args:
            z   : (B, D) latent tensor (zlocal)
            ctx : (B,) context labels (int64)

        Returns:
            z_centered : (B, D)
        """
        assert z.ndim == 2
        assert ctx.ndim == 1
        assert z.shape[0] == ctx.shape[0]

        device = z.device
        dtype = z.dtype

        z_centered = z.clone()

        # lazily initialize buffers
        if not hasattr(self, "_ctx_mean"):
            self._ctx_mean = {}
            self._ctx_count = {}

        unique_ctx = ctx.unique()
        # Subtract running mean for each context in batch
        for c in unique_ctx:
            mask = ctx == c
            c = int(c.item())
            if not mask.any():
                continue
            # Only update means during training
            if self.training:
                z_c = z[mask]                     # (Nc, D)
                batch_mean = z_c.mean(dim=0)      # (D,)
 
                # initialize if needed
                if c not in self._ctx_mean:
                    self._ctx_mean[c] = batch_mean.detach()
                    self._ctx_count[c] = 1
                else:
                    self._ctx_mean[c] = (
                        m * self._ctx_mean[c]
                        + (1.0 - m) * batch_mean.detach()
                    )
                    self._ctx_count[c] += 1

            # subtract running mean
            z_centered[mask] -= self._ctx_mean[c].to(device=device, dtype=dtype)

        return z_centered
    
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        ctrl_contrastive_weight: float = 0.0,
        adv_ctx_weight: float = 0.0,
        ctx_weight: float = 0.0,
        cls_weight: float = 0.0,
        cls_align_weight: float = 1.0,
        l_kl_weight: float = 0.0,
        decor_weight: float = 0.0,
        use_posterior_mean: bool = True,
        local_predictions: bool = True,
        graph_weight: float = 0.0,
        rerank_weight: float = 0.0,
        T_align: float | None = None,
        clip_k: int | None = None,
        n_unseen: int = 0,
        ctx_momentum: float = 0.98,
        top_k: int = 20,
        **kwargs,
    ) -> LossOutput:
        # ---- DEBUG ----
        self._step = self._step + 1
        """Compute the loss."""
        from torch.distributions import kl_divergence
        from src.losses import clip

        # Unpack input tensor
        x: torch.Tensor = inference_outputs[MODULE_KEYS.X_KEY]
        ctx: torch.Tensor = inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY]
        cls: torch.Tensor = inference_outputs[MODULE_KEYS.LABEL_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]
        g: torch.Tensor | None = tensors.get(REGISTRY_KEYS.GENE_EMB_KEY, None)

        # Compute basic kl divergence between prior and posterior x distributions
        kl_divergence_z_mat = kl_divergence(qz, pz)
        # Calculate reconstruction loss over batch and all features
        reconst_loss_mat = -px.log_prob(x)
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
            reconst_loss = reconst_loss_mat.mean(-1) * reconst_loss_mat.size(1)
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
        # Center z on control cell mectrl_class_idxan if control cells are encoded
        #z, cls, ctx = self._center_z_on_ctrl(z, cls, ctx)
        # Split batch into z and z control
        z, cls, ctx, contr_ctrl_loss = self._contr_ctrl_loss(z, cls, ctx)
        # Add contrastive loss to control
        if contr_ctrl_loss > 0 and ctrl_contrastive_weight:
            total_loss = total_loss + contr_ctrl_loss * ctrl_contrastive_weight
            extra_metrics[LOSS_KEYS.CTRL_CONTR_LOSS] = contr_ctrl_loss

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
            extra_outputs['ctx_labels'] = ctx_labels

        # Set empty classification logits
        logits = None

        # Use local latent space if available, else fall back to shared latent
        if 'qzl' in inference_outputs:
            _qz = inference_outputs['qzl']
            z_ = inference_outputs['zl']
        else:
            _qz = qz
            z_ = z
        # Base classification / clip losses on posterior mean or sampled latent space
        _z = _qz.loc if use_posterior_mean else z_
        # Center z
        _z = self._center_z(_z, ctx, m=ctx_momentum)
        
        # Calculate classification loss if positive weight
        if cls_weight > 0:
            z_cls_logits = self.classifier(_z)
            z_ce_loss = self._ce_cls_loss(z_cls_logits, y=cls)
            z_ce_loss = self._reduce_loss(z_ce_loss, reduction=self.non_elbo_reduction)
            # Add classification loss details
            lo_kwargs.update({
                'classification_loss': z_ce_loss,
                'logits': z_cls_logits,
            })
            # Add to total loss
            total_loss = total_loss + z_ce_loss * cls_weight
            # Set default logits
            logits = z_cls_logits

        # Add alignment loss if non-zero weight
        if io.non_zero(cls_align_weight):
            # Get class embedding transformer output
            cls_emb = self.cached_cls_emb
            # Local latent regularizations
            if 'qzl' in inference_outputs:
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
            # Do alignment forward pass and calculate losses
            alignment_output = self.aligner(
                x=_z, cls_emb=cls_emb, return_logits=False
            )
            
            # Extract shared latent space
            z_shared = alignment_output.get(MODULE_KEYS.Z_SHARED_KEY)
            # Extract the embedding projections to shared space
            cls2z = alignment_output.get(MODULE_KEYS.CLS_PROJ_KEY)
            
            # Choose alignment temperature
            T_align = T_align if T_align is not None else self.aligner.temperature
            # Log temperature
            extra_metrics['T_align'] = T_align
            
            # Adversial context loss
            if adv_ctx_weight > 0:
                L_adv_ctx, ctx_pred, ctx_labels = self.adversarial_context_loss(z_shared, ctx)
                total_loss = total_loss + adv_ctx_weight * L_adv_ctx
                extra_metrics['L_adv_ctx'] = L_adv_ctx
                extra_outputs['adv_ctx_pred'] = ctx_pred
                extra_outputs['adv_ctx_labels'] = ctx_labels
            
            # Get augmented aligment output
            if self.training and self.augmentation is not None:
                aug_align_out = self.get_augmented_alignment(
                    inference_outputs, cls_emb=cls_emb, T=T_align, g=g, use_posterior_mean=use_posterior_mean,
                )
                # Update clip loss inputs
                z_shared = torch.cat(
                    [z_shared, aug_align_out[MODULE_KEYS.Z_SHARED_KEY]], dim=0
                )
                cls = torch.cat((cls, aug_align_out[MODULE_KEYS.LABEL_KEY]))
                # Update loss kw args
                lo_kwargs['true_labels'] = cls
            
            # Calculate individual alignment losses
            alignment_losses: dict = clip.loss(
                z=z_shared,
                y=cls.flatten(),
                emb=cls2z,
                k=clip_k,
                use_reverse=False,
                T=T_align,
                n_unseen=n_unseen,
                training=self.training,
                reduction=self.non_elbo_reduction
            )

            # Calculate class similarities
            _cls_emb: torch.Tensor = cls2z[:self.n_labels] if local_predictions else cls2z
            logits = clip.clip_logits_z2c(z_shared, _cls_emb, T=T_align)

            # Add combined alignment loss to final loss
            align_loss = alignment_losses.pop(LOSS_KEYS.LOSS)
            
            # Log individual alignment losses
            extra_metrics.update(alignment_losses)
            
            # Add projected embedding representations to extra outputs
            extra_outputs[MODULE_KEYS.Z_SHARED_KEY] = z_shared
            extra_outputs[MODULE_KEYS.CLS_PROJ_KEY] = cls2z
                        
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
            
            # Add embedding graph regularizer
            if graph_weight > 0.0:
                L_graph = self.laplacian_smoothness_loss(cls2z)
                total_loss = total_loss + graph_weight * L_graph
                extra_metrics['L_graph'] = L_graph
            
            # Add top-k predictions loss
            if rerank_weight > 0.0:
                top_k_predictions = self.re_rank_loss(z_shared, cls2z, logits, cls.flatten(), K=top_k)
                L_rank = top_k_predictions.pop(LOSS_KEYS.LOSS)
                total_loss = total_loss + rerank_weight * L_rank
                extra_metrics.update(top_k_predictions)
                extra_outputs[PREDICTION_KEYS.PREDICTION_KEY] = self.top_k_predict(z_shared, cls2z, logits)
                
            # Add prototype regularizer
            if cls2z.ndim == 3:
                L_prot_reg = self.cls_emb.prototype_div_reg(cls2z)
                total_loss = total_loss + L_prot_reg
                extra_metrics['L_prot_reg'] = L_prot_reg
        else:
            alignment_output = None
            
        # Only return predictions of observable classes
        if logits is not None and PREDICTION_KEYS.PREDICTION_KEY not in extra_outputs:
            # Add class predictions to extra output
            extra_outputs[PREDICTION_KEYS.PREDICTION_KEY] = logits.argmax(dim=-1).flatten().detach()
        # Add logits to extra output
        extra_outputs[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = logits
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
