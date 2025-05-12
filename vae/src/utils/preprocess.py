from anndata import AnnData
import numpy as np
import logging
from src.utils.constants import REGISTRY_KEYS


def _prep_adata(
            adata: AnnData, 
            cls_labels: list[str], 
            ctrl_key: str = 'control', 
            p_pos: int = -1,
            verbose: bool = True,
        ) -> None:
        # Add control label
        adata.obs['is_ctrl'] = False
        adata.obs.loc[adata.obs[cls_labels[p_pos]]==ctrl_key, 'is_ctrl'] = True
        # Count number of cells per cls_labels
        cls_range = np.arange(len(cls_labels))
        mask = cls_range != cls_range[p_pos]
        cls_labels_excl = np.array(cls_labels)[mask].tolist()
        cpp = adata.obs.groupby(cls_labels_excl, observed=True)['is_ctrl'].value_counts()
        # Get
        idc = []
        ctrl_idc = []
        for group, data in adata.obs[~adata.obs.is_ctrl].groupby(cls_labels_excl, observed=True):
            mask = True
            idc.extend(data.index)
            for i, g in enumerate(group):
                mask &= adata.obs[cls_labels[i]]==g
            n_p = cpp[*group, False]
            if verbose:
                logging.info(f'group: {group}: n_ctrl: {np.sum(mask)}, n_pert: {n_p}')
            ctrl_idc.extend(adata.obs[mask].sample(n_p).index)
        # Extract control cells
        ctrl_layer = adata[ctrl_idc].X.copy()
        # Filter cells for perturbed cells only
        adata._inplace_subset_obs(idc)
        # Set basal layer with unperturbed cells
        adata.layers[REGISTRY_KEYS.B_KEY] = ctrl_layer
