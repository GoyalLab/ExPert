from typing import Iterable

from src.modules._base import (
    Encoder, 
    DecoderSCVI, 
    ContextClassAligner, 
    Classifier,
    ClassEmbedding
)
import src.utils.io as io
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS, LOSS_KEYS, PREDICTION_KEYS
import src.utils.embeddings as emb_utils
from src.utils.common import grad_reverse
from src.utils.augmentations import BatchAugmentation

from typing import Iterable

import torch
import torch.nn.functional as F

from scvi.data import _constants
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.module.base import (
    LossOutput,
    auto_move_data,
)
from scvi.module._vae import VAE

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
    _align_ext_emb_loss_strategies = ['kl', 'clip']
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
        use_adapter: bool = True,
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
        decode_covariates: bool = True,
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
        use_feature_mask: bool = False,
        drop_prob: float = 1e-3,
        use_ctx_adv_cls: bool = True,
        use_hierachical_labels: bool = True,
        use_pretrained_emb_for_hierachy: bool = False,
        module_quantile: float = 0.9,
        pathway_quantile: float = 0.7,
        use_augmentation: bool = True,
        extra_encoder_kwargs: dict | None = {},
        extra_decoder_kwargs: dict | None = {},
        extra_ctx_adv_classifier_kwargs: dict | None = {},
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
            encode_covariates=False,
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

        # Setup external embeddings
        self.ctx_emb = torch.nn.Embedding.from_pretrained(ctx_emb, freeze=True)
        if use_adapter:
            extra_cls_emb_kwargs['n_output'] = n_latent
        self.pretrained_emb = cls_emb
        self.cls_emb = ClassEmbedding(pretrained_emb=cls_emb, class_texts=cls_text_dict, **extra_cls_emb_kwargs)
        self.cls_sim = torch.nn.Embedding.from_pretrained(cls_sim, freeze=True) if cls_sim is not None else None
        # Hierarchical clustering parameters
        self.module_quantile = module_quantile
        self.pathway_quantile = pathway_quantile
        self.cls2module = None  # Will be set on first reset_cached_cls_emb
        self.cls2pw = None
        self.module_weight = None
        self.pathway_weight = None
        # Setup embedding params
        self.n_ctx, self.n_ctx_dim = ctx_emb.shape
        self.n_cls, self.n_cls_dim = self.cls_emb.shape
        self.use_joint = False
        # Setup class proxies and hierachical labels
        self.use_hierachical_labels = use_hierachical_labels
        self.use_pretrained_emb_for_hierachy = use_pretrained_emb_for_hierachy
        self.gene_names = list(cls_text_dict.keys()) if cls_text_dict is not None else None
        self.reset_cached_cls_emb(verbose=True, reset_labels=True)
        # Save if we have unseen embeddings or not
        self.has_unseen_ctx = self.n_ctx > n_batch
        self.has_unseen_cls = self.n_cls > n_labels
        self.n_shared = n_shared if n_shared is not None else n_latent
        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'

        # ----- Setup encoder module -----
        self.default_encoder_kwargs = {
            'n_input': n_input,
            'n_output': n_latent,
            'n_hidden': n_hidden,
            'n_cat_list': None,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm_encoder,
            'use_layer_norm': use_layer_norm_encoder,
            'var_activation': var_activation,
            'use_feature_mask': use_feature_mask,
            'drop_prob': drop_prob,
            'n_dim_context_emb': self.n_ctx_dim,
        }
        extra_encoder_kwargs.update(self.default_encoder_kwargs)
        self.z_encoder = Encoder(**extra_encoder_kwargs)

        # ----- Setup decoder module -----
        # Covariate params
        self.decode_covariates = decode_covariates
        self.decode_context_projection = decode_context_projection
        # Whether to decode covariates
        n_input_decoder = n_latent + n_continuous_cov * decode_covariates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        decoder_cat_list = cat_list if decode_covariates else None
        n_ctx_dim_decoder = None
        # Whether to include covariates in decoder
        if decode_covariates:
            # Use projected context embedding
            if decode_context_projection:
                n_ctx_dim_decoder = self.n_shared
            # Use raw context embedding
            else:
                n_ctx_dim_decoder = self.n_ctx_dim
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

        # Setup adversial context classifier
        if use_ctx_adv_cls:
            self.ctx_adv_classifier = Classifier(
                n_input=n_latent,
                n_labels=self.n_batch,
                **extra_ctx_adv_classifier_kwargs,
            )
        else:
            self.ctx_adv_classifier = None

        # ----- Setup external embedding aligner -----

        # Setup an additional classifier on z, purely supervised
        self.classifier = Classifier(
            n_input=n_latent,
            n_labels=self.n_labels,
            **extra_cls_kwargs,
        )
       
        # Setup aligner
        self.aligner = ContextClassAligner(
            n_input=n_latent,
            n_shared=self.n_shared,
            ctx_emb_dim=self.n_ctx_dim,
            cls_emb_dim=self.n_cls_dim,
            **extra_aligner_kwargs
        )

        # Check loss strategies
        self.set_align_ext_emb_strategies(align_ext_emb_loss_strategy)

        # ----- Debug -----
        self._step = 0
        
    def reset_cached_cls_emb(self, verbose: bool = False, reset_labels: bool = False):
        """Reset embedding on epoch start"""
        # Recompute gene embeddings
        with torch.no_grad():
            self.cached_cls_emb = self.cls_emb()
        # Set embeddings for labels to either cached or pretrained
        if self.use_pretrained_emb_for_hierachy:
            cls_emb = self.pretrained_emb.to(self.cached_cls_emb.device)
        else:
            cls_emb = self.cached_cls_emb
        # Reset hierarchical labels
        if reset_labels and self.use_hierachical_labels:
            self._set_hierarchical_labels(cls_emb, verbose=verbose)
        
    def _set_hierarchical_labels(
        self, 
        cls_emb: torch.Tensor, 
        mode: str = 'library',
        verbose: bool = False
    ):
        """Set module and pathway labels with adaptive thresholding"""
        n = self.n_cls
        # Set hierarchy based on class embedding similarities
        if mode == 'embedding':
            self._set_hierarchy_embedding(cls_emb)
        elif mode == 'library':
            self._set_hierarchical_labels_from_database()
        
        # Set unseen indices for all layers
        self.unseen_gene_indices = torch.arange(self.n_labels, n, device=cls_emb.device)
        # Get modules/pathways for observed genes
        observed_modules = set(self.cls2module[:self.n_labels].tolist())
        observed_pathways = set(self.cls2pw[:self.n_labels].tolist())
        
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
            
    def _set_hierarchical_labels_from_database(
        self, 
        n_pathways_per_module: int = 3,
        library: str = 'Reactome_2022', 
        verbose: bool = False
    ):
        """
        Use biological pathways instead of clustering
        """
        import gseapy as gp
        
        gene_names = self.gene_names
        n = len(gene_names)
        
        # Get pathway database
        try:
            gene_sets = gp.get_library(library, organism='Human')
        except:
            log.warning(f"Failed to download {library}, using KEGG...")
            gene_sets = gp.get_library('KEGG_2021_Human', organism='Human')
        
        # Build gene → pathways mapping
        gene_to_pathways = {}
        pathway_to_genes = {}
        
        for pathway_name, genes_in_pathway in gene_sets.items():
            for gene in genes_in_pathway:
                if gene in gene_names:
                    if gene not in gene_to_pathways:
                        gene_to_pathways[gene] = []
                    gene_to_pathways[gene].append(pathway_name)
                    
                    if pathway_name not in pathway_to_genes:
                        pathway_to_genes[pathway_name] = []
                    pathway_to_genes[pathway_name].append(gene)
        
        # Assign each gene to primary pathway
        pathway_assignments = []
        for gene in gene_names:
            if gene in gene_to_pathways and len(gene_to_pathways[gene]) > 0:
                # Take most specific pathway (shortest name, or first alphabetically)
                pathways = gene_to_pathways[gene]
                pathway = min(pathways, key=lambda x: (len(x), x))
            else:
                pathway = "Unknown_Pathway"
            pathway_assignments.append(pathway)
        
        # Convert to indices
        unique_pathways = sorted(set(pathway_assignments))
        pathway_to_idx = {p: i for i, p in enumerate(unique_pathways)}
        
        self.cls2pw = torch.tensor(
            [pathway_to_idx[p] for p in pathway_assignments],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create modules: genes with multiple shared pathways
        module_assignments = []
        for gene in gene_names:
            if gene in gene_to_pathways and len(gene_to_pathways[gene]) > 0:
                # Module = combination of pathways
                pathways = sorted(gene_to_pathways[gene][:n_pathways_per_module])  # Top N pathways
                module = "__".join(pathways)
            else:
                module = "Unknown_Module"
            module_assignments.append(module)
        
        unique_modules = sorted(set(module_assignments))
        module_to_idx = {m: i for i, m in enumerate(unique_modules)}
        
        self.cls2module = torch.tensor(
            [module_to_idx[m] for m in module_assignments],
            device=self.cls2pw.device
        )
        
        if verbose:
            log.info(f"Hierarchical labels from biological databases:")
            log.info(f"  {n} genes")
            log.info(f"  {len(unique_modules)} modules")
            log.info(f"  {len(unique_pathways)} pathways")
            
            # Show distribution
            pw_unique, pw_counts = self.cls2pw.unique(return_counts=True)
            mod_unique, mod_counts = self.cls2module.unique(return_counts=True)
            
            log.info(f"  Pathway distribution:")
            log.info(f"    Min size: {pw_counts.min()}, Max size: {pw_counts.max()}, Mean: {pw_counts.float().mean():.1f}")
            log.info(f"  Module distribution:")
            log.info(f"    Min size: {mod_counts.min()}, Max size: {mod_counts.max()}, Mean: {mod_counts.float().mean():.1f}")
            
            # Show top pathways
            top_pathways_idx = pw_counts.argsort(descending=True)[:5]
            log.info(f"  Top 5 pathways:")
            for idx in top_pathways_idx:
                pathway_name = unique_pathways[pw_unique[idx].item()]
                count = pw_counts[idx].item()
                log.info(f"    - {pathway_name}: {count} genes")
            
    def _set_hierarchy_embedding(self, cls_emb: torch.Tensor):
        cls_emb_norm = F.normalize(cls_emb, dim=-1)
        similarity = cls_emb_norm @ cls_emb_norm.T
        
        # Remove diagonal
        n = similarity.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
        sim_flat = similarity[mask]
        
        # Module: mean + k*std (tight clustering)
        module_threshold = torch.quantile(sim_flat, self.module_quantile)
        
        # Pathway: mean + k*std (loose clustering)  
        pathway_threshold = torch.quantile(sim_flat, self.pathway_quantile)
        
        module_adj = (similarity > module_threshold).float()
        self.cls2module = self._connected_components(module_adj)
        
        pathway_adj = (similarity > pathway_threshold).float()
        self.cls2pw = self._connected_components(pathway_adj)

    def _connected_components(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Find connected components in adjacency matrix (simple greedy assignment)"""
        n = adj_matrix.size(0)
        labels = torch.arange(n, device=adj_matrix.device)
        
        # Union-find style: merge connected nodes
        for i in range(n):
            neighbors = torch.where(adj_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                min_label = labels[neighbors].min()
                labels[neighbors] = min_label
                labels[i] = min_label
        
        # Compress labels to 0, 1, 2, ...
        unique_labels = labels.unique(sorted=True)
        label_map = {old.item(): new for new, old in enumerate(unique_labels)}
        compressed = torch.tensor([label_map[l.item()] for l in labels], 
                                device=labels.device)
        
        return compressed
    
    def _get_module_embeddings(self, cls2z: torch.Tensor) -> torch.Tensor:
        """Aggregate gene embeddings into module embeddings"""
        cls2module = self.cls2module[:cls2z.size(0)]
        n_modules = cls2module.max().item() + 1
        module_embs = torch.zeros(n_modules, cls2z.size(1), device=cls2z.device)
        
        for i in range(n_modules):
            mask = (cls2module == i)
            if mask.any():
                module_embs[i] = cls2z[mask].mean(dim=0)
        
        return F.normalize(module_embs, dim=-1)

    def _get_pathway_embeddings(self, cls2z: torch.Tensor) -> torch.Tensor:
        """Aggregate gene embeddings into pathway embeddings"""
        cls2pw = self.cls2pw[:cls2z.size(0)]
        n_pathways = cls2pw.max().item() + 1
        pathway_embs = torch.zeros(n_pathways, cls2z.size(1), device=cls2z.device)
        
        for i in range(n_pathways):
            mask = (cls2pw == i)
            if mask.any():
                pathway_embs[i] = cls2z[mask].mean(dim=0)
        
        return F.normalize(pathway_embs, dim=-1)
    
    def _get_unseen_label_idx(self):
        return getattr(self, 'unseen_gene_indices')
    
    def _get_unseen_module_idx(self):
        return getattr(self, 'unseen_module_indices')
    
    def _get_unseen_pathway_idx(self):
        return getattr(self, 'unseen_pathway_indices')

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
        
    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        if full_forward_pass or self.minified_data_type is None:
            loader = "full_data"
        elif self.minified_data_type in [
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR_WITH_COUNTS,
        ]:
            loader = "minified_data"
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")
        # Do full forward pass
        if loader == "full_data":
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.LABEL_KEY: tensors[REGISTRY_KEYS.LABELS_KEY],
                MODULE_KEYS.G_EMB_KEY: tensors.get(REGISTRY_KEYS.GENE_EMB_KEY, None),
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            }
        # Perform a cached forward passed with minified model
        else:
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }
    
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
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x_.sum(1)).unsqueeze(1)
        # Apply CMP normalization to x
        if self.use_cpm:
            x_ = x_ / library * 1e6
        # Apply log1p transformation to input batch
        if self.log_variational:
            x_ = torch.log1p(x_)
        # TODO: add min-max scaling

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # Add context embedding
        ctx_emb = self.ctx_emb.weight
        # Perform forward pass through encoder
        inference_out: dict[str, torch.Tensor] = self.z_encoder(encoder_input, *categorical_input, g=g, context_emb=ctx_emb)
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
        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
            REGISTRY_KEYS.LABELS_KEY: label,
            REGISTRY_KEYS.BATCH_KEY: batch_index
        }
    
    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        # Return generative data
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: inference_outputs[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.Y_KEY: inference_outputs[REGISTRY_KEYS.LABELS_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),
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

        from scvi.distributions import (
            NegativeBinomial,
            Normal,
            Poisson,
            ZeroInflatedNegativeBinomial,
        )

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
        if self.decode_covariates:
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
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        # Aligner forward pass using either qz mean or sampled z
        _z = qz.loc if use_posterior_mean else z
        
        # Optionally use different embeddings, fall back to internals if none are given
        if ctx_emb is None:
            ctx_emb = self.ctx_emb.weight
        if cls_emb is None:
            cls_emb = self.cached_cls_emb
            
        # Add z classifier output
        z_cls_logits = self.classifier(_z)
        inference_outputs['logits'] = z_cls_logits
        # Use regular alignment
        if not self.use_joint:
            align_out: dict[str, torch.Tensor] = self.aligner(
                _z, 
                ctx_emb=ctx_emb, cls_emb=cls_emb,
                ctx_idx=batch_index, cls_idx=label,
                return_logits=return_logits,
                ctrl_idx=self.ctrl_class_idx
            )
        else:
            # Classify using a joint embedding of context and class
            align_out: dict[str, torch.Tensor] = self.aligner.classify(
                _z, ctx_emb=ctx_emb, cls_emb=cls_emb
            )
            
        # Update prediction output
        logits = self.predict(
            z_shared=align_out[MODULE_KEYS.Z_SHARED_KEY],
            cls2z=align_out[MODULE_KEYS.CLS_PROJ_KEY],
            return_logits=True,
            **kwargs
        )
        align_out[MODULE_KEYS.CLS_LOGITS_KEY] = logits
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
        if self.use_hierachical_labels and io.non_zero(module_temp) and io.non_zero(pathway_temp):
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
        # Apply reduction
        if loss is None:
            return None
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'batchmean':
            return loss.sum(-1).mean()
        else:
            raise ValueError(f'Invalid reduction: {reduction}')
    
    def _clip_loss_simple(
            self, 
            z: torch.Tensor, 
            y: torch.Tensor, 
            emb: torch.Tensor, 
            T: torch.Tensor,
            return_full: bool = True
        ) -> torch.Tensor:
        # Ensure y is a vector
        y = y.flatten()
        # Normalize z and embedding
        z = F.normalize(z, dim=-1)

        # For class -> latent, restrict to the classes present in the batch
        chosen = F.normalize(emb[y], dim=-1)   # (B, d)
        # Each class embedding should match its corresponding latent (diagonal)
        labels = torch.arange(z.size(0), device=z.device)

        # Latent -> class loss
        logits_z2c = (z @ chosen.T) / T     # (B, B)
        loss_z2c = F.cross_entropy(logits_z2c, labels, reduction='none')
        # Class -> latent loss
        logits_c2z = (chosen @ z.T) / T         # (B, B)
        loss_c2z = F.cross_entropy(logits_c2z, labels, reduction='none')
        # Symmetric loss per sample
        if return_full:
            return 0.5 * (loss_z2c + loss_c2z)
        else:
            return loss_z2c, loss_c2z
        
    def clip_loss_v2(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        emb: torch.Tensor,
        T: torch.Tensor,
        k: int = 10,
        unique_proxies: bool = True,
        return_full: bool = True,
        use_reverse: bool = True,
    ):
        """
        U-CLIP loss with unique-class prototypes per batch + hard negative mining.
        
        Args:
            z: (B, d) - cell embeddings from scRNA encoder
            y: (B,) - class labels (which gene was perturbed)
            emb: (C, d) - full class embedding matrix (all gene descriptions)
            T: temperature parameter
            k: number of hard negatives to mine (None or 0 for full softmax)
            unique_proxies: if True, only use unique classes in batch (U-CLIP)
            return_full: if True, return combined loss; if False, return tuple
            use_reverse: if True, compute symmetric class->cell loss
        
        Returns:
            loss: scalar if return_full=True
            (loss_z2c, loss_c2z): tuple if return_full=False
        """
        device = z.device
        y = y.flatten()
        B = z.size(0)
        
        # ---------------------------------------------------------
        # Normalize latent and class embeddings
        # ---------------------------------------------------------
        z = F.normalize(z, dim=-1)
        
        # ---------------------------------------------------------
        # Unique-class CLIP (U-CLIP)
        # ---------------------------------------------------------
        if unique_proxies:
            unique_classes, inv = torch.unique(y, return_inverse=True)
            # U = number of unique classes in the batch
            proxies = F.normalize(emb[unique_classes], dim=-1)  # (U, d)
            # positive target index is inv (mapping sample -> unique prototype)
            targets = inv
        else:
            # Use per-sample embeddings (no uniqueness)
            proxies = F.normalize(emb[y], dim=-1)  # (B, d)
            targets = torch.arange(B, device=device)
        
        U = proxies.size(0)
        
        # ==========================================================
        # Forward direction: z -> class (cell -> gene description)
        # ==========================================================
        logits_z2c = (z @ proxies.T) / T  # (B, U)
        
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
                logits_z2c_mined,
                torch.zeros(B, dtype=torch.long, device=device),
                reduction='none',
            )
        else:
            # Standard full-softmax over all U classes
            loss_z2c = F.cross_entropy(logits_z2c, targets, reduction='none')
        
        # ==========================================================
        # Reverse direction: class -> z (gene description -> cell)
        # ==========================================================
        if use_reverse:
            if unique_proxies:
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
                                logits_mined,
                                torch.zeros(num_pos, dtype=torch.long, device=device),
                                reduction='none',
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
                chosen = F.normalize(emb[y], dim=-1)  # (B, d)
                logits_c2z = (chosen @ z.T) / T  # (B, B)
                loss_c2z = F.cross_entropy(logits_c2z, labels, reduction='none')
        else:
            # No reverse loss
            loss_c2z = torch.zeros(1, device=device, requires_grad=True)
        
        # ---------------------------------------------------------
        # Combine losses
        # ---------------------------------------------------------
        if return_full:
            if use_reverse:
                # Symmetric: average both directions
                return 0.5 * (
                    self._reduce_loss(loss_z2c, self.non_elbo_reduction) + 
                    self._reduce_loss(loss_c2z, self.non_elbo_reduction)
                )
            else:
                # Asymmetric: only forward direction
                return self._reduce_loss(loss_z2c, self.non_elbo_reduction)
        else:
            # Return both components (useful for logging/analysis)
            return loss_z2c, loss_c2z
    
    def _joint_clip_loss(
        self,
        z: torch.Tensor,
        joint_emb: torch.Tensor,
        T: float | None = 0.1,
    ) -> torch.Tensor:
        """Calculate joint embedding clip loss based on both batch-specific embeddings."""
        # Get temperature default if none
        if T is None:
            T = self.aligner.joint_temperature
        # Normalize latent space
        z = F.normalize(z, dim=-1)

        # Normalize embeddings
        b_joint_emb = F.normalize(joint_emb, dim=-1)

        # --- Compute symmetric CLIP loss ---
        # (1) Latent --> Joint
        logits_z2joint = (z @ b_joint_emb.T) / T
        labels = torch.arange(z.size(0), device=z.device)
        loss_z2joint = F.cross_entropy(logits_z2joint, labels, reduction="none")

        # (2) Joint --> Latent
        logits_joint2z = (b_joint_emb @ z.T) / T
        loss_joint2z = F.cross_entropy(logits_joint2z, labels, reduction="none")

        # Symmetric loss
        return 0.5 * (loss_z2joint + loss_joint2z)
    
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
        logits_rev = grad_reverse(logits, lam)
        p = torch.softmax(logits_rev, dim=-1)
        return (p * p.log()).sum(dim=-1).mean()
    
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

    def _pseudo_latent_loss(
        self,
        z: torch.Tensor,
        frac: float = 0.1,
        alpha: float = 0.0,
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ensure that a random latent alignment still has an unbiased distribution over all possible classes."""
        B, D = z.shape
        B_pseudo = int(B * frac)
        device = z.device
        # Sample random latents
        z_pseudo = torch.randn(B_pseudo, D, device=device)
        # Revert gradient on pseudo z
        #pseudo_z_rev = grad_reverse(z_pseudo)
        # Sample random context and class indices
        ctx_idx = torch.randint(0, self.n_ctx, (B_pseudo,), device=z.device)
        cls_idx = torch.randint(0, self.n_cls, (B_pseudo,), device=z.device)
        # Align pseudo-latent to embeddings
        align_out = self.aligner(
            z_pseudo,
            ctx_emb=self.ctx_emb.weight, cls_emb=self.cached_cls_emb,
            ctx_idx=ctx_idx, cls_idx=cls_idx,
            return_logits=True,
            T=T,
        )
        # Extract logits from pseudo-alignment
        z_pseudo_shared: torch.Tensor = align_out[MODULE_KEYS.Z_SHARED_KEY]
        p_ctx_logits: torch.Tensor = align_out[MODULE_KEYS.CTX_LOGITS_KEY]
        p_cls_logits: torch.Tensor = align_out[MODULE_KEYS.CLS_LOGITS_KEY]

        # Save buffers of argmax predictions for epoch-level loss and monitoring during training
        self._update_pseudo_buffer(p_ctx_logits, self._ctx_key)
        self._update_pseudo_buffer(p_cls_logits, self._cls_key)

        # Calculate BCE on prediciting observed class over unobserved
        ctx_loss = self._bias_bce(p_ctx_logits, n_obs=self.n_batch, n_cls=self.n_ctx)
        cls_loss = self._bias_bce(p_cls_logits, n_obs=self.n_labels, n_cls=self.n_cls)
        return alpha * ctx_loss + (1-alpha) * cls_loss
    
    def _manifold_regularization_loss(
        self,
        ctx_proj: torch.Tensor,
        cls_proj: torch.Tensor,
        frac: float = 0.1,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Calculate manifold regularization loss to ensure projected embedding keeps geometry intact."""
        ctx_reg_loss = torch.tensor(0.0)
        if self.has_unseen_ctx:
            n_ctx_sample = int(frac * self.n_ctx)
            ctx_reg_loss = emb_utils.manifold_regularization(ext_emb=self.ctx_emb.weight, proj_emb=ctx_proj, n_sample=n_ctx_sample)
        cls_reg_loss = torch.tensor(0.0)
        if self.has_unseen_cls:
            n_cls_sample = int(frac * self.n_cls)
            cls_reg_loss = emb_utils.manifold_regularization(ext_emb=self.cached_cls_emb, proj_emb=cls_proj, n_sample=n_cls_sample)
        # Compute final regularization loss
        return alpha * ctx_reg_loss + (1-alpha) * cls_reg_loss
    
    def _ce_cls_loss(self, logits: torch.Tensor, y: torch.Tensor, T: float | None = None) -> torch.Tensor:
        """Calculate additional simple classification loss on initial latent space."""
        # Get classifier temperature
        if T is None:
            T = self.classifier.temperature
        # Return an CE loss
        loss = F.cross_entropy(logits, y, reduction='none')
        return loss
    
    def _adv_ctx_loss(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Calculate adversial context classification loss on z with reversed gradients."""
        z_rev = grad_reverse(z)
        adv_ctx_logits = self.ctx_adv_classifier(z_rev)
        # Calculate loss
        return F.cross_entropy(adv_ctx_logits, ctx.squeeze(-1), reduction='none')
    
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
    
    def clip_loss(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        emb: torch.Tensor,
        T: torch.Tensor,
        k: int = 10,
        unique_proxies: bool = True,
        reduce: bool = True,
        return_logits: bool = False,
        use_reverse: bool = True,
        n_unseen: int = 10,
        unseen_weight: float = 0.3,
        unseen_indices: torch.Tensor | None = None
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
        
        # Normalize proxies
        proxies = F.normalize(proxies, dim=-1)
        U = proxies.size(0)
        
        # Forward direction: z -> class (cell -> gene description)
        logits_z2c = (z @ proxies.T) / T  # (B, U)
        
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
                logits_z2c_mined,
                torch.zeros(B, dtype=torch.long, device=device),
                reduction='none',
            )
        else:
            # Standard full-softmax over all U classes
            loss_z2c = F.cross_entropy(logits_z2c, targets, reduction='none')
        
        # UNSEEN NEGATIVES: Add unseen gene embeddings as hard negatives
        loss_z2c_unseen = None
        if use_unseen_negatives and len(unseen_indices) > 0 and n_unseen > 0:
            # Sample random unseen embeddings
            n_unseen_eff = min(n_unseen, len(unseen_indices))
            sampled_unseen_indices = unseen_indices[torch.randperm(len(unseen_indices), device=device)[:n_unseen_eff]]
            unseen_sample = emb_norm[sampled_unseen_indices]  # (n_unseen, d)
            
            # Augmented class embeddings: [seen_proxies, unseen_proxies]
            augmented_proxies = torch.cat([proxies, unseen_sample], dim=0)  # (U + n_unseen, d)
            
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
                    logits_z2c_aug_mined,
                    torch.zeros(B, dtype=torch.long, device=device),
                    reduction='none',
                )
            else:
                # Full softmax with augmented classes
                loss_z2c_unseen = F.cross_entropy(
                    logits_z2c_aug, 
                    targets, 
                    reduction='none'
                )
        
        # Reverse direction: class -> z (gene description -> cell)
        logits_c2z = None
        if use_reverse:
            if unique_proxies:
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
                                logits_mined,
                                torch.zeros(num_pos, dtype=torch.long, device=device),
                                reduction='none',
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
                loss_c2z = F.cross_entropy(logits_c2z, labels, reduction='none')
        else:
            # No reverse loss
            loss_c2z = torch.zeros(1, device=device, requires_grad=True)
        
        # ---------------------------------------------------------
        # Combine losses
        # ---------------------------------------------------------
        o = {}
        # Add logits to output
        if return_logits:
            o.update({
                LOSS_KEYS.Z2C_LOGITS: logits_z2c,
                LOSS_KEYS.C2Z_LOGITS: logits_c2z
            })
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
        # Return a single loss value
        if reduce and not return_logits:
            return o[LOSS_KEYS.LOSS]
        # Return full output
        return o

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        ctrl_contrastive_weight: float = 0.0,
        cls_weight: float = 0.0,
        ctx_align_weight: float = 0.0,
        cls_align_weight: float = 0.0,
        joint_align_weight: float = 0.0,
        use_posterior_mean: bool = False,
        random_unseen_replacement_p: float = 0.0,
        pseudo_latent_frac: float = 0.0,
        manifold_regularization_frac: float = 0.0,
        pseudo_latent_weight: float = 0.0,
        manifold_regularization_weight: float = 0.0,
        align_temp_reg_weight: float = 0.0,
        joint_alpha: float = 0.8,
        T_align: float | None = None,
        hard_negatives: int = 0,
        n_unseen: int = 10,
        unseen_weight: float = 0.3,
        use_reverse: bool = True,
        local_predictions: bool = True,
        # Hierachical parameters
        module_weight: float = 0.3,
        module_t: float = 0.4,
        pw_weight: float = 0.2,
        pw_t: float = 0.8,
        **kwargs,
    ) -> LossOutput:
        # ---- DEBUG ----
        self._step = self._step + 1
        """Compute the loss."""
        from torch.distributions import kl_divergence

        # Unpack input tensor
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        ctx: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]
        cls: torch.Tensor = tensors[REGISTRY_KEYS.LABELS_KEY]
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
            reconst_loss = reconst_loss_mat.mean(-1)
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
                    reconst_loss[ctrl_mask] = 0.0
                    kl_divergence_z[ctrl_mask] = 0.0
                elif not self.use_reconstruction_control:
                    reconst_loss[ctrl_mask] = 0.0
                    kl_weight = kl_weight * ctrl_frac
                elif not self.use_kl_control:
                    kl_divergence_z[ctrl_mask] = 0.0
                    rl_weight = rl_weight * ctrl_frac
                else:
                    # Use full batch
                    n_obs_minibatch = x.shape[0]
        
        # Weighted reconstruction and KL
        weighted_reconst_loss = rl_weight * reconst_loss
        weighted_kl_local = kl_weight * kl_divergence_z

        # Save reconstruction losses
        kl_locals = {MODULE_KEYS.KL_Z_KEY: kl_divergence_z}
        # Batch normalize elbo loss of reconstruction + kl --> batchmean reduction
        total_loss = torch.mean(weighted_reconst_loss + weighted_kl_local)

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

        # Collect extra outputs
        extra_outputs = {}

        # Re-format labels
        ctx = ctx.squeeze(-1)
        cls = cls.squeeze(-1)

        # Calculate classification loss on z
        if cls_weight > 0:
            z_cls_logits = self.classifier(z)
            z_ce_loss = self._ce_cls_loss(z_cls_logits, y=cls)
            z_ce_loss = self._reduce_loss(z_ce_loss, reduction=self.non_elbo_reduction)
            # Add classification loss details
            lo_kwargs.update({
                'classification_loss': z_ce_loss,
                'logits': z_cls_logits,
            })
            # Add to total loss
            total_loss = total_loss + z_ce_loss * cls_weight

        # Only do alignment part if loss > 0
        predictions = None
        cls_t = 1.0
        if io.non_zero(ctx_align_weight) or io.non_zero(cls_align_weight) or io.non_zero(joint_align_weight):
            # Get class embedding transformer output
            cls_emb = self.cached_cls_emb
            # Align to posterior mean of latent space or full distribution
            _z = qz.loc if use_posterior_mean else z
            # Do alignment
            alignment_output: dict = self.aligner(
                _z, 
                ctx_emb=self.ctx_emb.weight,
                ctx_idx=ctx,
                cls_emb=cls_emb,
                cls_idx=cls,
                T=T_align,
                alpha=joint_alpha,
            )
            # Randomly set some labels to unseen contexts and classes to keep these embeddings active
            if random_unseen_replacement_p > 0:
                ctx, cls = self._random_unseen_replacement(ctx_idx=ctx.unsqueeze(0), cls_idx=cls.unsqueeze(0), p=random_unseen_replacement_p)
            # Extract shared latent space
            z_shared = alignment_output.get(MODULE_KEYS.Z_SHARED_KEY)
            # Extract the embedding projections to shared space
            ctx2z = alignment_output.get(MODULE_KEYS.CTX_PROJ_KEY)
            cls2z = alignment_output.get(MODULE_KEYS.CLS_PROJ_KEY)
            # Extract joint embedding projection
            joint2z = alignment_output.get(MODULE_KEYS.JOINT_PROJ_KEY)
            # Add projected embedding representations to extra outputs
            extra_outputs[MODULE_KEYS.Z_SHARED_KEY] = z_shared
            extra_outputs[MODULE_KEYS.CLS_PROJ_KEY] = cls2z
            # Extract context logits from aligner
            ctx_logits = alignment_output.get(MODULE_KEYS.CTX_LOGITS_KEY)
            if ctx_logits is None:
                # Calculate manually
                ctx_logits = self.aligner.get_ctx_logits(z_shared, ctx2z, T=T_align)
            # Extract class logits from aligner
            cls_logits = alignment_output.get(MODULE_KEYS.CLS_LOGITS_KEY)
            
            # Either do individual clips or on joint embedding
            if joint_align_weight > 0:
                # Mark that we used joint embedding to align
                self.use_joint = True
                # Align to combined embedding space
                joint_loss_ps = self._joint_clip_loss(
                    _z,
                    joint_emb=joint2z,
                    T=T_align
                )
                joint_loss = self._reduce_loss(joint_loss_ps, reduction=self.non_elbo_reduction)
                # Add to extra metrics
                extra_metrics[LOSS_KEYS.JOINT_LOSS] = joint_loss
                # Compare z to batch joint embeddings
                cls_out = self.aligner.classify(
                    _z,
                    ctx_emb=self.ctx_emb.weight[:self.n_batch],
                    cls_emb=cls_emb[:self.n_labels],
                    T=T_align,
                    alpha=joint_alpha,
                )
                # Get full logits
                joint_logits = cls_out[MODULE_KEYS.JOINT_LOGITS_KEY]
                logits = cls_out[MODULE_KEYS.CLS_LOGITS_KEY]
                # Add to overall alignment loss
                align_loss = joint_loss * joint_align_weight
            else:
                # Calculate clip loss for context embedding
                ctx_loss_ps = self.clip_loss(
                    z_shared, 
                    y=ctx.flatten(), 
                    emb=ctx2z, 
                    T=self.aligner.ctx_temperature,
                    k=hard_negatives,
                    n_unseen=0,
                    unseen_weight=0.0
                )
                # Augment data batch
                if self.augmentation:
                    x_, batch_index, label, cont_covs, cat_covs = self.augmentation(
                        **self._get_inference_input(tensors)
                    )
                    # Do second forward pass through the vae encoder and update z_shared
                    z_shared_clip = self._regular_inference(x_, batch_index, label, g=g, cont_covs=cont_covs, cat_covs=cat_covs)[MODULE_KEYS.Z_KEY]
                else:
                    z_shared_clip, label = z_shared, cls.flatten()
                # Get temperature for class clip
                if T_align is not None:
                    cls_t = T_align
                else:
                    cls_t = self.aligner.cls_temperature

                # Get unseen indices
                # Calculate clip loss for class embedding
                cls_clip_loss_out = self.clip_loss(
                    z_shared_clip, 
                    y=label.flatten(), 
                    emb=cls2z, 
                    T=cls_t,
                    k=hard_negatives,
                    use_reverse=use_reverse,
                    n_unseen=n_unseen,
                    unseen_weight=unseen_weight,
                    unseen_indices=self._get_unseen_label_idx(),
                    reduce=False,
                    return_logits=True,
                )
                # Extract losses
                loss_z2c_ps = cls_clip_loss_out[LOSS_KEYS.Z2C]
                loss_c2z_ps = cls_clip_loss_out[LOSS_KEYS.C2Z]
                loss_z2c_unseen_ps = cls_clip_loss_out[LOSS_KEYS.UNSEEN_Z2C]
                cls_logits = cls_clip_loss_out[LOSS_KEYS.Z2C_LOGITS]
                # Combine losses
                loss_z2c = self._reduce_loss(loss_z2c_ps, reduction=self.non_elbo_reduction)
                loss_c2z = self._reduce_loss(loss_c2z_ps, reduction=self.non_elbo_reduction)
                loss_z2c_unseen = self._reduce_loss(loss_z2c_unseen_ps, reduction=self.non_elbo_reduction)
                cls_loss = 0.5 * (loss_z2c + loss_c2z)
                # Add unseen loss if not None
                if loss_z2c_unseen is not None:
                    cls_loss = (1-unseen_weight) * cls_loss + unseen_weight * loss_z2c_unseen
                    extra_metrics[f'z2c_loss_unseen'] = loss_z2c_unseen
                # Apply reductions
                ctx_loss = self._reduce_loss(ctx_loss_ps, reduction=self.non_elbo_reduction)
                # Add to extra metrics
                extra_metrics[LOSS_KEYS.CLIP_CTX_CLS_LOSS] = ctx_loss
                extra_metrics[LOSS_KEYS.CLIP_CLS_LOSS] = cls_loss
                # Log individual parts of clip loss
                extra_metrics[f'z2c_loss'] = loss_z2c
                extra_metrics[f'c2z_loss'] = loss_c2z
                # Add hierachical losses
                hierachical_losses = []
                ex_weight_sum = torch.tensor(0.0)
                # Module level, Medium temperature
                if module_weight > 0 and module_t is not None and module_t > 0 and self.use_hierachical_labels:
                    # Use hierachical module labels instead of genes
                    loss_module = self.clip_loss(
                        z_shared, 
                        y=self.cls2module[cls.flatten()], 
                        emb=self._get_module_embeddings(cls2z), 
                        T=cls_t,
                        k=0,                                 # Don't use hard negatives here
                        use_reverse=False,                   # Reverse can't be applied to aggregated proxies
                        n_unseen=n_unseen,
                        unseen_weight=unseen_weight,
                        unseen_indices=self._get_unseen_module_idx(),
                    )
                    # Get reduced loss
                    hierachical_losses.append(loss_module * module_weight)
                    # Add to logging
                    extra_metrics['module_loss'] = loss_module
                    # Update weights
                    ex_weight_sum = ex_weight_sum + module_weight
                    # Update instance weight
                    self.module_weight = module_weight
                # Pathway level, Highest temperature
                if pw_weight > 0 and pw_t is not None and pw_t > 0 and self.use_hierachical_labels:
                    # Use hierachical pathway labels instead of genes
                    loss_pw = self.clip_loss(
                        z_shared, 
                        y=self.cls2pw[cls.flatten()],
                        emb=self._get_pathway_embeddings(cls2z), 
                        T=cls_t,
                        k=0,                                 # Don't use hard negatives here
                        use_reverse=False,                   # Reverse can't be applied to aggregated proxies
                        n_unseen=n_unseen,
                        unseen_weight=unseen_weight,
                        unseen_indices=self._get_unseen_pathway_idx(),
                    )
                    # Get reduced loss
                    hierachical_losses.append(loss_pw * pw_weight)
                    extra_metrics['pathway_loss'] = loss_pw
                    ex_weight_sum = ex_weight_sum + pw_weight
                    # Update instance weight
                    self.pathway_weight = pw_weight
                
                # Update classification loss
                if len(hierachical_losses) > 0:
                    # Get remaining weight for gene loss
                    h_cls_w = 1 - ex_weight_sum
                    hierachical_losses.append(cls_loss * h_cls_w)
                    # Combine cls losses
                    cls_loss = torch.stack(hierachical_losses).mean()
                
                # Set logits to class embedding logits
                logits = cls_logits
                
                # Combine alignment losses
                align_loss = ctx_loss * ctx_align_weight + cls_loss * cls_align_weight
            
            # Get predictions
            logits = self.predict(
                z_shared=z_shared, cls2z=cls2z,
                gene_temp=cls_t, module_temp=cls_t, pathway_temp=cls_t, 
                return_logits=True
            )
            # Only return predictions of observable classes
            if local_predictions:
                logits = logits[:,:self.n_labels]
            # Add class predictions to extra output
            extra_outputs[PREDICTION_KEYS.PREDICTION_KEY] = logits.squeeze(-1).detach()
            
            # Add additional regularization losses
            if pseudo_latent_frac > 0 and pseudo_latent_weight > 0:
                # Enforce model to not be biased towards observed classes
                pseudo_latent_loss = self._pseudo_latent_loss(_z, frac=pseudo_latent_frac, T=T_align)
                # Only add loss if not nan
                if not torch.isnan(pseudo_latent_loss):
                    # Add to overall alignment loss
                    align_loss = align_loss + pseudo_latent_loss * pseudo_latent_weight
                    # Add to extra metrics for logging
                    extra_metrics[LOSS_KEYS.PSEUDO_Z_LOSS] = pseudo_latent_loss
            # Add a manifold regularization loss on all embedding projections
            if manifold_regularization_frac > 0:
                mr_loss = self._manifold_regularization_loss(ctx_proj=ctx2z, cls_proj=cls2z, frac=manifold_regularization_frac, alpha=0.5)
                # Add to overall alignment loss
                align_loss = align_loss + mr_loss * manifold_regularization_weight
                # Add to extra metrics
                extra_metrics[LOSS_KEYS.MANIFOLD_REG_LOSS] = mr_loss
            # Add a small regularization on temperature
            if align_temp_reg_weight > 0:
                temp_reg_loss = self.aligner.get_temp_reg_loss()
                align_loss = align_loss + temp_reg_loss * align_temp_reg_weight
                extra_metrics[LOSS_KEYS.T_REG_LOSS] = temp_reg_loss
            # Add to extra metrics
            extra_metrics[LOSS_KEYS.ALIGN_LOSS] = align_loss
            # Add to total loss
            total_loss = total_loss + align_loss
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
