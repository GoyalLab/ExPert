import gseapy as gp
import torch
import torch.nn.functional as F
import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import logging
from typing import Literal


class GeneAnnotation:
    
    def __init__(
        self,
        gene_emb: torch.Tensor | None,
        gene_names: list[str] | None,
        libs: list[str] = ["Reactome_2022", "KEGG_2021_Human"],
        min_genes_per_pathway: int = 5,
        source_weights: dict[str, float] | None = None,
        pathway_frac: float = 0.6,
        module_frac: float = 0.8,
        method: Literal['hierarchical', 'graph'] | None = None,
        verbose: bool = True,
        metric: str = 'cosine',
        device: str = 'cuda',
        plot: bool = True,
        k: int = 10,
    ):
        # Parameter check
        if gene_emb is None and gene_names is None:
            raise ValueError('Provide either gene embedding or gene names.')
        self.libs = libs
        self.min_genes_per_pathway = min_genes_per_pathway
        self.source_weights = source_weights
        self.pathway_frac = pathway_frac
        self.module_frac = module_frac
        self.method = method
        self.verbose = verbose
        self.device = device
        self.plot = plot
        # Either calculate gene embedding based on annotations or register provided
        self.gene_names = gene_names
        self.gene_emb = gene_emb
        # Overwrite gene embedding if it is None
        if gene_emb is None:
            self._set_annotation_gene_embedding()
            self.metric = 'jaccard'
            self.annotation_based = True
        else:
            self.metric = metric
            self.valid_mask = None
            self.annotation_based = False
        # Set dimensionality of gene embedding
        self.n_cls = self.gene_emb.shape[0]
        self.n_dim = self.gene_emb.shape[1]
        # Set empty indices
        self.misc_pw_idx = -1
        self.misc_mod_idx = -1
        # Set hierarchy
        if self.method is not None:
            self._calculate_hierarchy()
        # Build nearest neighbor graph
        self.laplacian_graph = self.get_laplacian_graph(k=k)
         
    def get_cls2pw(self):
        return getattr(self, 'cls2pw', None)
    
    def get_cls2module(self):
        return getattr(self, 'cls2module', None)
    
    def get_module2pw(self):
        return getattr(self, 'module2pw', None)
        
    def _set_annotation_gene_embedding(self) -> torch.Tensor:
        """
        Build hierarchical grouping:
            Reactome top-level (pathway) -> sub-pathway/module -> gene

        Args
        ----
        genes : list[str]
            List of gene symbols used as "classes" in your model.
        reactome_lib : str
            gseapy library name for Reactome.
        min_genes_per_pathway : int
            Filter out tiny pathways.
        Returns
        -------
        matrix (genes x pathways)
        """
        # Get list of genes
        genes = list(self.gene_names)
        gene_set = set(genes)
        C = len(genes)
        gene2idx = {g: i for i, g in enumerate(genes)}

        # If no external weights are provided, weight each source equally
        if self.source_weights is None:
            self.source_weights = {lib: 1.0 for lib in self.libs}
        # Collect data blocks and names
        feature_blocks = []
        feature_names = []  # (lib, set_name)
        # Collect each library information
        for lib in self.libs:
            if self.verbose:
                logging.info(f"Loading library: {lib}")
            # Load library
            try:
                gs = gp.get_library(name=lib)
            except Exception as e:
                logging.warning(e)
                continue

            # Filter gene sets for provided gene names
            sets = []
            for set_name, set_genes in gs.items():
                overlap = gene_set.intersection(set_genes)
                if len(overlap) >= self.min_genes_per_pathway:
                    sets.append((set_name, sorted(overlap)))
            if self.verbose:
                logging.info(f" - Retained {len(sets)} sets after filtering.")
            # Disregard data if no intersecting gene sets were found
            if len(sets) == 0:
                continue

            # build block: (C, num_sets_lib)
            P_lib = len(sets)
            # Start with an empty matrix
            X_lib = torch.zeros((C, P_lib), dtype=torch.float32)
            # Add pathway information from each source
            for j, (set_name, g_list) in enumerate(sets):
                # Set binary weights
                for g in g_list:
                    i = gene2idx[g]
                    X_lib[i, j] = 1.0
                # Collect names
                feature_names.append((lib, set_name))

            # Apply source weight
            X_lib *= self.source_weights.get(lib, 1.0)

            feature_blocks.append(X_lib)
        # Check if any information was found
        if len(feature_blocks) == 0:
            raise ValueError("No overlapping gene sets found in any library.")

        # Concatenate all features horizontally
        X = torch.cat(feature_blocks, dim=1)  # (C, total_features)
        
        # Create mask of genes that don't have any pathway information
        valid_mask = X.sum(-1) > 0
        n_valid = valid_mask.sum()
        n_missing = X.shape[0] - n_valid
        # Check for misc genes
        if self.verbose and n_missing > 0:
            logging.info(f'Found {n_missing} genes with no annotations, creating a miscellanious category.')
        # Set class parameters
        self.gene_emb = X.to(self.device)
        self.valid_mask = valid_mask
        
    def _calculate_hierarchy(self):
        if self.method == "hierarchical":
            return self._calculate_hierarchy_hierarchical()
        elif self.method == "graph":
            return self._calculate_hierarchy_graph()
        else:
            raise ValueError(f"Unknown hierarchy method: {self.method}")

    def _calculate_hierarchy_hierarchical(self):
        cls_emb = self.gene_emb
        if self.valid_mask is not None:
            cls_emb = cls_emb[self.valid_mask]

        if self.metric == "cosine":
            cls_emb = F.normalize(cls_emb, dim=-1)

        emb_np = cls_emb.detach().cpu().numpy()

        # Pairwise distances
        dist_condensed = pdist(emb_np, metric=self.metric)
        self.Z = linkage(dist_condensed, method="average")

        if self.plot:
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram
            plt.figure(figsize=(25, 10))
            dendrogram(self.Z)
            plt.show()
            plt.close()

        C_valid = cls_emb.shape[0]
        n_pathways = max(2, int(C_valid * self.pathway_frac))
        n_modules  = max(n_pathways + 1, int(C_valid * self.module_frac))

        pw_labels  = fcluster(self.Z, t=n_pathways, criterion="maxclust") - 1
        mod_labels = fcluster(self.Z, t=n_modules,  criterion="maxclust") - 1

        return self._finalize_hierarchy(pw_labels, mod_labels)
    
    @torch.no_grad()
    def get_laplacian_graph(
        self,
        k: int = 10,
    ):
        """
        Returns edge_index (2, E_edges) and edge_weight (E_edges,)
        representing sparse adjacency A.
        """
        E = self.gene_emb
        E = E.to(self.device)

        # Calculate similarity
        E_n = F.normalize(E, dim=-1)
        S = E_n @ E_n.T  # (C, C)

        # Get top-k similarities
        vals, idx = torch.topk(S, k=k+1, dim=1)
        A = torch.zeros_like(S)
        A.scatter_(1, idx[:, 1:], vals[:, 1:])  # skip self
        # Make A symmetric
        A = (A + A.T) / 2
        # Get laplacian graph
        D = torch.diag(A.sum(dim=1))
        L = D - A
        return L
    
    def _calculate_hierarchy_graph(self):
        import igraph as ig
        import leidenalg
        from sklearn.metrics import pairwise_distances

        cls_emb = self.gene_emb
        if self.valid_mask is not None:
            cls_emb = cls_emb[self.valid_mask]

        if self.metric == "cosine":
            cls_emb = F.normalize(cls_emb, dim=-1)

        emb_np = cls_emb.detach().cpu().numpy()

        # Similarity matrix
        if self.metric == "cosine":
            S = emb_np @ emb_np.T
        else:
            D = pairwise_distances(emb_np, metric=self.metric)
            S = 1.0 - D
        
        # Set diagonal to 0
        np.fill_diagonal(S, 0)
        
        # Threshold weak edges
        threshold = getattr(self, "pw_threshold", 0.6)
        edges = np.where(S > threshold)
        weights = S[edges]

        g = ig.Graph(
            n=emb_np.shape[0],
            edges=list(zip(edges[0], edges[1])),
            edge_attrs={"weight": weights.tolist()},
        )
        g.simplify(combine_edges="max", loops=True)

        # Coarse = pathways
        pw_part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=getattr(self, "pathway_resolution", 2.0),
        )
        
        # Choose higher treshold for modules
        threshold = getattr(self, "mod_threshold", 0.8)
        edges = np.where(S > threshold)
        weights = S[edges]
        
        g = ig.Graph(
            n=emb_np.shape[0],
            edges=list(zip(edges[0], edges[1])),
            edge_attrs={"weight": weights.tolist()},
        )
        g.simplify(combine_edges="max", loops=True)

        # Fine = modules
        mod_part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=getattr(self, "module_resolution", 3.0),
        )

        pw_labels  = np.array(pw_part.membership)
        mod_labels = np.array(mod_part.membership)

        return self._finalize_hierarchy(pw_labels, mod_labels)
    
    def _finalize_hierarchy(self, pw_labels, mod_labels):
        device = self.device

        # -------------------------------------
        # Build cls2pw / cls2module
        # -------------------------------------
        if self.valid_mask is None:
            cls2pw     = torch.from_numpy(pw_labels).long()
            cls2module = torch.from_numpy(mod_labels).long()
        else:
            C = self.gene_emb.shape[0]
            cls2pw     = torch.full((C,), -1, dtype=torch.long)
            cls2module = torch.full((C,), -1, dtype=torch.long)

            cls2pw[self.valid_mask]     = torch.from_numpy(pw_labels).long()
            cls2module[self.valid_mask] = torch.from_numpy(mod_labels).long()

        # -------------------------------------
        # Identify singleton modules
        # -------------------------------------
        uniq_mod, mod_counts = cls2module.unique(return_counts=True)
        singleton_modules = uniq_mod[mod_counts == 1]

        for m in singleton_modules:
            cls2module[cls2module == m] = -1

        # -------------------------------------
        # Identify singleton pathways
        # -------------------------------------
        uniq_pw, pw_counts = cls2pw.unique(return_counts=True)
        singleton_pathways = uniq_pw[pw_counts == 1]

        for p in singleton_pathways:
            cls2pw[cls2pw == p] = -1

        # -------------------------------------
        # Reindex surviving module labels
        # -------------------------------------
        valid_mods = cls2module >= 0
        mod_ids = cls2module[valid_mods].unique(sorted=True)
        mod_map = {int(old): i for i, old in enumerate(mod_ids)}

        for old, new in mod_map.items():
            cls2module[cls2module == old] = new

        M = len(mod_ids)

        # -------------------------------------
        # Reindex surviving pathway labels
        # -------------------------------------
        valid_pws = cls2pw >= 0
        pw_ids = cls2pw[valid_pws].unique(sorted=True)
        pw_map = {int(old): i for i, old in enumerate(pw_ids)}

        for old, new in pw_map.items():
            cls2pw[cls2pw == old] = new

        P = len(pw_ids)

        # -------------------------------------
        # Assign misc indices
        # -------------------------------------
        self.misc_mod_idx = M
        self.misc_pw_idx  = P

        cls2module[cls2module < 0] = self.misc_mod_idx
        cls2pw[cls2pw < 0]         = self.misc_pw_idx

        # -------------------------------------
        # Build module --> pathway mapping
        # -------------------------------------
        module2pw = torch.full((M + 1,), self.misc_pw_idx, dtype=torch.long)

        for m in range(M):
            mask = cls2module == m
            if mask.any():
                pws = cls2pw[mask]
                vals, counts = pws.unique(return_counts=True)
                module2pw[m] = vals[counts.argmax()]

        # -------------------------------------
        # Move to device
        # -------------------------------------
        self.cls2pw     = cls2pw.to(device)
        self.cls2module = cls2module.to(device)
        self.module2pw  = module2pw.to(device)
