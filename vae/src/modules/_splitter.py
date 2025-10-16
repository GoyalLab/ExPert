import lightning.pytorch as pl
import numpy as np
from typing import Literal
from src.data._contrastive_loader import ContrastiveAnnDataLoader
from sklearn.model_selection import train_test_split

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.dataloaders._semi_dataloader import SemiSupervisedDataLoader
from scvi.dataloaders._data_splitting import validate_data_split, validate_data_split_with_external_indexing


class SemiSupervisedDataSplitter(pl.LightningDataModule):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.
    The ratio between labeled and unlabeled data in adata will be preserved
    in the train/test/val sets.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is None, which is practicaly 0.9 and potentially adding small last
        batch to validation cells)
    validation_size
        float, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and test set
        are split in the sequential order of the data according to `validation_size` and
        `train_size` percentages.
    n_samples_per_label
        Number of subsamples for each label class to sample per epoch
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~scvi.data.AnnDataLoader`.
    external_indexing
        A list of data split indices in the order of training, validation, and test sets.
        Validation and test set are not required and can be left empty.
        Note that per group (train,valid,test) it will cover both the labeled and unlebeled parts
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~scvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~scvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.SCVI.setup_anndata(adata, labels_key="labels")
    >>> adata_manager = scvi.model.SCVI(adata).adata_manager
    >>> unknown_label = "label_0"
    >>> splitter = SemiSupervisedDataSplitter(adata, unknown_label)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        n_samples_per_label: int | None = None,
        pin_memory: bool = False,
        external_indexing: list[np.array, np.array, np.array] | None = None,
        shuffle_train: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size_is_none = not bool(train_size)
        self.train_size = 0.9 if self.train_size_is_none else float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.drop_last = kwargs.pop("drop_last", False)
        self.data_loader_kwargs = kwargs
        self.n_samples_per_label = n_samples_per_label
        self.shuffle_train = shuffle_train

        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()
        self.unlabeled_category = labels_state_registry.unlabeled_category
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category).ravel()

        self.pin_memory = pin_memory
        self.external_indexing = external_indexing

    def setup(self, stage: str | None = None):
        """Split indices in train/test/val sets."""
        n_labeled_idx = len(self._labeled_indices)
        n_unlabeled_idx = len(self._unlabeled_indices)

        if n_labeled_idx != 0:
            # Need to separate to the external and non-external cases of the labeled indices
            if self.external_indexing is not None:
                # first we need to intersect the external indexing given with the labeled indices
                labeled_idx_train, labeled_idx_val, labeled_idx_test = (
                    np.intersect1d(self.external_indexing[n], self._labeled_indices)
                    for n in range(3)
                )
                n_labeled_train, n_labeled_val = validate_data_split_with_external_indexing(
                    n_labeled_idx,
                    [labeled_idx_train, labeled_idx_val, labeled_idx_test],
                    self.data_loader_kwargs.pop("batch_size", settings.batch_size),
                    self.drop_last,
                )
            else:
                n_labeled_train, n_labeled_val = validate_data_split(
                    n_labeled_idx,
                    self.train_size,
                    self.validation_size,
                    self.data_loader_kwargs.pop("batch_size", settings.batch_size),
                    self.drop_last,
                    self.train_size_is_none,
                )

                labeled_permutation = self._labeled_indices
                if self.shuffle_set_split:
                    rs = np.random.RandomState(seed=settings.seed)
                    labeled_permutation = rs.choice(
                        self._labeled_indices, len(self._labeled_indices), replace=False
                    )

                labeled_idx_val = labeled_permutation[:n_labeled_val]
                labeled_idx_train = labeled_permutation[
                    n_labeled_val : (n_labeled_val + n_labeled_train)
                ]
                labeled_idx_test = labeled_permutation[(n_labeled_val + n_labeled_train) :]
        else:
            labeled_idx_test = []
            labeled_idx_train = []
            labeled_idx_val = []

        if n_unlabeled_idx != 0:
            # Need to separate to the external and non-external cases of the unlabeled indices
            if self.external_indexing is not None:
                # we need to intersect the external indexing given with the labeled indices
                unlabeled_idx_train, unlabeled_idx_val, unlabeled_idx_test = (
                    np.intersect1d(self.external_indexing[n], self._unlabeled_indices)
                    for n in range(3)
                )
                n_unlabeled_train, n_unlabeled_val = validate_data_split_with_external_indexing(
                    n_unlabeled_idx,
                    [unlabeled_idx_train, unlabeled_idx_val, unlabeled_idx_test],
                    self.data_loader_kwargs.pop("batch_size", settings.batch_size),
                    self.drop_last,
                )
            else:
                n_unlabeled_train, n_unlabeled_val = validate_data_split(
                    n_unlabeled_idx,
                    self.train_size,
                    self.validation_size,
                    self.data_loader_kwargs.pop("batch_size", settings.batch_size),
                    self.drop_last,
                    self.train_size_is_none,
                )

                unlabeled_permutation = self._unlabeled_indices
                if self.shuffle_set_split:
                    rs = np.random.RandomState(seed=settings.seed)
                    unlabeled_permutation = rs.choice(
                        self._unlabeled_indices,
                        len(self._unlabeled_indices),
                        replace=False,
                    )

                unlabeled_idx_val = unlabeled_permutation[:n_unlabeled_val]
                unlabeled_idx_train = unlabeled_permutation[
                    n_unlabeled_val : (n_unlabeled_val + n_unlabeled_train)
                ]
                unlabeled_idx_test = unlabeled_permutation[(n_unlabeled_val + n_unlabeled_train) :]
        else:
            unlabeled_idx_train = []
            unlabeled_idx_val = []
            unlabeled_idx_test = []

        indices_train = np.concatenate((labeled_idx_train, unlabeled_idx_train))
        indices_val = np.concatenate((labeled_idx_val, unlabeled_idx_val))
        indices_test = np.concatenate((labeled_idx_test, unlabeled_idx_test))

        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)

        if len(self._labeled_indices) != 0:
            self.data_loader_class = SemiSupervisedDataLoader
            dl_kwargs = {
                "n_samples_per_label": self.n_samples_per_label,
            }
        else:
            self.data_loader_class = AnnDataLoader
            dl_kwargs = {}

        self.data_loader_kwargs.update(dl_kwargs)

    def train_dataloader(self):
        """Create the train data loader."""
        return self.data_loader_class(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create the validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create the test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass


class ContrastiveDataSplitter(pl.LightningDataModule):
    """
    Contrastive data splitter. Creates label and context balanced batches.
    """
    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        max_cells_per_batch: int = 24,
        max_classes_per_batch: int = 20,
        ctrl_class: str | None = None,
        last_first: bool = True,
        shuffle_classes: bool = True,
        pin_memory: bool = False,
        use_contrastive_loader: Literal['train', 'val', 'both'] | None = 'train',
        use_control: Literal['train', 'val', 'both'] | None = 'train',
        external_indexing: list[np.array, np.array, np.array] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size_is_none = not bool(train_size)
        self.train_size = 0.9 if self.train_size_is_none else float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.drop_last = kwargs.pop("drop_last", False)
        self.data_loader_kwargs = kwargs
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self.ctrl_class = ctrl_class
        self.last_first = last_first
        self.shuffle_classes = shuffle_classes
        self.use_contrastive_loader = use_contrastive_loader
        self.use_control = use_control
        # Fall back to product of max cells per batch * max classes per batch
        _bs = self.data_loader_kwargs.get("batch_size")
        if _bs is None:
            self.batch_size = int(max_cells_per_batch * max_classes_per_batch)
            self.data_loader_kwargs["batch_size"] = self.batch_size
        else:
            self.batch_size = _bs
        # Set indices, labels, and batch labels
        self.indices = range(adata_manager.adata.n_obs)
        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()
        batch_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        batches = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.batch.attr_name,
            batch_state_registry.original_key,
        ).ravel()
        self.labels = labels
        self.batches = batches
        self.pin_memory = pin_memory
        self.external_indexing = external_indexing

    def setup(self, stage: str | None = None):
        """Split indices in train/test/val sets."""
        # Split dataset into train and validation set
        train_idx, val_idx = train_test_split(
            self.indices,
            train_size=self.train_size,
            stratify=self.labels,           # Preserve class distribution
            shuffle=self.shuffle_set_split,
            random_state=settings.seed
        )
        
        # Split indices are stratified, TODO: add test split logic
        indices_train = np.array(train_idx)
        indices_val = np.array(val_idx)
        indices_test = np.array([])

        # Don't validate on control cells if we have any registered
        if self.ctrl_class is not None and self.use_control not in ['val', 'both']:
            ctrl_mask = self.labels == self.ctrl_class
            ctrl_indices = np.array(self.indices)[ctrl_mask]

            # Only keep non-control validation indices
            indices_val = np.setdiff1d(indices_val, ctrl_indices, assume_unique=True)

        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)

        self.train_labels = self.labels[self.train_idx]
        self.train_batches = self.batches[self.train_idx]
        self.val_labels = self.labels[self.val_idx]
        self.val_batches = self.batches[self.val_idx]

        # Setup train data loader
        self.data_loader_class = AnnDataLoader
        # Set some defaults for data loader class
        self.train_data_loader_kwargs = self.data_loader_kwargs.copy()
        # Use contrastive loader for training
        if self.use_contrastive_loader in ['train', 'both']:
            self.data_loader_class = ContrastiveAnnDataLoader
            self.train_data_loader_kwargs.update({
                'labels': self.train_labels,
                'batches': self.train_batches,
                'max_cells_per_batch': self.max_cells_per_batch,
                'max_classes_per_batch': self.max_classes_per_batch,
                'ctrl_class': self.ctrl_class
            })
        # Set validation loader
        self.val_loader_class = AnnDataLoader
        self.val_data_loader_kwargs = self.data_loader_kwargs.copy()
        # Use contrastive loader for validation
        if self.use_contrastive_loader in ['val', 'both']:
            self.val_loader_class = ContrastiveAnnDataLoader
            self.val_data_loader_kwargs.update({
                'labels': self.val_labels,
                'batches': self.val_batches,
                'max_cells_per_batch': self.max_cells_per_batch,
                'max_classes_per_batch': self.max_classes_per_batch,
                'ctrl_class': self.ctrl_class
            })

    def train_dataloader(self):
        """Create the train data loader."""
        return self.data_loader_class(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=self.shuffle_classes,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **self.train_data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create the validation data loader."""
        if len(self.val_idx) > 0:
            return self.val_loader_class(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.val_data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create the test data loader."""
        if len(self.test_idx) > 0:
            return self.val_loader_class(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
