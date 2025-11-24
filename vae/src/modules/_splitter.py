import lightning.pytorch as pl
import numpy as np
from typing import Literal
from sklearn.model_selection import train_test_split

from src.data import BalancedAnnDataLoader
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._ann_dataloader import AnnDataLoader

import logging
log = logging.getLogger(__name__)


class DataSplitter(pl.LightningDataModule):
    """
    General data splitter supporting both supervised and contrastive training setups.

    loader_type: {'supervised', 'contrastive', 'vanilla'}
        Controls which data loader is used.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        loader_type: Literal['balanced', 'vanilla'] = 'balanced',
        train_size: float | None = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        n_samples: int | None = 500,
        batch_size: int = 512,
        ctrl_class: str | None = None,
        ctrl_frac: float | None = 1.0,
        last_first: bool = True,
        shuffle_classes: bool = True,
        pin_memory: bool = False,
        use_copy: bool = True,
        use_special_for_split: list[str] = ['train'],
        test_context_labels: np.ndarray | None = None,
        external_indexing: list[np.ndarray, np.ndarray, np.ndarray] | None = None,
        cache_indices: dict[str, np.ndarray] | None = None,
        drop_last: bool = False,
        use_balanced_weights: bool = True,
        use_contrastive_loader: bool = True,
        n_classes_per_batch: int | None = None,
        n_samples_per_class: int | None = None,
        min_contexts_per_class: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.loader_type = loader_type
        self.train_size_is_none = not bool(train_size)
        self.train_size = 0.9 if self.train_size_is_none else float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.drop_last = drop_last
        self.data_loader_kwargs = kwargs

        # loader-specific parameters
        self.n_samples = n_samples * batch_size
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.min_contexts_per_class = min_contexts_per_class
        self.use_balanced_weights = use_balanced_weights
        self.use_contrastive_loader = use_contrastive_loader
        
        self.batch_size = batch_size
        self.ctrl_class = ctrl_class
        self.ctrl_frac = ctrl_frac
        self.last_first = last_first
        self.shuffle_classes = shuffle_classes
        self.pin_memory = pin_memory
        self.use_copy = use_copy
        self.test_context_labels = test_context_labels
        self.external_indexing = external_indexing
        self.cache_indices = cache_indices
        self.use_special_for_split = use_special_for_split

        # === Label and context extraction ===
        self.indices = np.arange(adata_manager.adata.n_obs)
        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()
        batch_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        self.batches = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.batch.attr_name,
            batch_state_registry.original_key,
        ).ravel()

    def setup(self, stage: str | None = None):
        """Split indices in train/test/val sets."""
        # === Split logic (identical to contrastive) ===
        if self.cache_indices is None:
            if self.test_context_labels is not None:
                test_mask = np.isin(self.batches, self.test_context_labels)
                base_idx = self.indices[~test_mask]
                test_idx = self.indices[test_mask]
            else:
                base_idx = self.indices
                test_idx = []

            # Stratified split for reproducibility
            class_labels = self.labels[base_idx]
            train_idx, val_idx = train_test_split(
                base_idx,
                train_size=self.train_size,
                stratify=class_labels,
                shuffle=self.shuffle_set_split,
                random_state=settings.seed,
            )

            indices_train = np.array(train_idx)
            indices_val = np.array(val_idx)
            indices_test = np.array(test_idx)
        else:
            indices_train = np.array(self.cache_indices.get("train"))
            indices_val = np.array(self.cache_indices.get("val"))
            indices_test = np.array(self.cache_indices.get("test"))

        # === Assign splits ===
        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)

        # === Subset metadata ===
        self.train_labels = self.labels[self.train_idx]
        self.train_batches = self.batches[self.train_idx]
        self.val_labels = self.labels[self.val_idx]
        self.val_batches = self.batches[self.val_idx]
        self.test_labels = self.labels[self.test_idx] if len(self.test_idx) else None
        self.test_batches = self.batches[self.test_idx] if len(self.test_idx) else None

        # === Loader selection === use AnnDataLoader as fallback
        self.loader_class_train = AnnDataLoader
        self.loader_class_val = AnnDataLoader
        self.loader_class_test = AnnDataLoader

        # Assign special loader types
        if "train" in self.use_special_for_split:
            if self.loader_type == "balanced":
                self.loader_class_train = BalancedAnnDataLoader
        if "val" in self.use_special_for_split:
            if self.loader_type == "balanced":
                self.loader_class_val = BalancedAnnDataLoader
                
    def get_base_kwargs(self, mode: str):
        if mode == "train":
            indices = self.train_idx
        elif mode == "val":
            indices = self.val_idx
        elif mode == "test":
            indices = self.test_idx
        else:
            raise ValueError(f"Invalid split {mode}")
        return {
            "indices": indices,
            "shuffle": self.shuffle_classes,
            "drop_last": self.drop_last,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size,
            **self.data_loader_kwargs,
        }
         
    def get_special_kwargs(self, mode: str):
        if mode == "train":
            labels = self.train_labels
            batches = self.train_batches
        elif mode == "val":
            labels = self.val_labels
            batches = self.val_batches
        elif mode == "test":
            labels = self.test_labels
            batches = self.test_batches
        else:
            raise ValueError(f"Invalid split {mode}")
        return {
            "labels": labels,
            "batches": batches,
            "use_balanced_weights": self.use_balanced_weights,
            "n_classes_per_batch": self.n_classes_per_batch,
            "n_samples_per_class": self.n_samples_per_class,
            "min_contexts_per_class": self.min_contexts_per_class,
            "use_contrastive_loader": self.use_contrastive_loader,
        }
        
    def _get_dataloader(self, mode: str):
        """Create dataloader of specific split.

        Args:
            mode (str): Data split, one of train, val, test

        Returns:
            DataLoader class
        """
        kwargs = self.get_base_kwargs(mode)
        if mode in self.use_special_for_split:
            if self.loader_type != "vanilla":
                kwargs.update(self.get_special_kwargs(mode))    
        # Only shuffle train dataloader
        if mode != "train":
            kwargs["shuffle"] = False
        # Select dataloader class
        if mode == "train":
            dataloader_cls = self.loader_class_train
        elif mode == "val":
            dataloader_cls = self.loader_class_val
        elif mode == "test":
            dataloader_cls = self.loader_class_test
        else:
            raise ValueError(f"Invalid split {mode}")
        return dataloader_cls(self.adata_manager, **kwargs)

    # === Dataloaders ===
    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        if len(self.val_idx) == 0:
            return None
        return self._get_dataloader('val')

    def test_dataloader(self):
        if len(self.test_idx) == 0:
            return None
        return self._get_dataloader('test')
