from copy import deepcopy
import numpy as np
import pandas as pd
import logging


class ContrastiveBatchSampler:
    IDX_KEY: str = 'indices'
    CLASSES_PER_BATCH_KEY: str = 'cbp'
    IDC_PER_BATCH_KEY: str = 'ipb'
    CLASS_POOL_INFO_KEY: str = 'cpi'

    def _check_setup(self) -> None:
        # None of the parameters are set, fall back to default split
        if self.max_classes_per_batch is None and self.max_cells_per_batch is None:
            logging.warning(f'Falling back to default max_classes_per_batch=8.')
            self.max_classes_per_batch = 8
            self.max_cells_per_batch = self.batch_size // self.max_classes_per_batch
        # One of the parameters is set, adjust other
        elif self.max_cells_per_batch is not None and self.max_classes_per_batch is None:
            self.max_classes_per_batch = self.batch_size // self.max_cells_per_batch
        elif self.max_classes_per_batch is not None and self.max_cells_per_batch is None:
            self.max_cells_per_batch = self.batch_size // self.max_classes_per_batch
        # Both are set, change nothing
        if self.batch_size % self.max_classes_per_batch != 0:
            logging.warning(f'Batch size is not directly divisible by {self.max_classes_per_batch}. This could lead to performance issues.')

    def __init__(
        self,
        indices: np.ndarray,
        cls_labels: np.ndarray,
        batch_size: int = 480,
        max_cells_per_batch: int | None = 24,
        max_classes_per_batch: int | None = 20,
        last_first: bool = True,
        shuffle_classes: bool = True,
    ):
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self._check_setup()
        self.shuffle_classes = shuffle_classes
        # Setup indices and labels
        self.indices = indices
        self.n_indices = len(indices)
        self.cls_labels = pd.Series(cls_labels)
        # Count number of cells per class
        self.cells_per_class = self.cls_labels.value_counts()
        # Define number of unique classes in data
        self.n_classes = self.cells_per_class.shape[0]
        # Determine number of batches to pick from indices with this batch size
        self.n_batches = np.floor(self.n_indices / self.batch_size)
        # Define pool of all indices per class to choose from
        self.idx_pools = [set(self.indices[self.cls_labels==label]) for label in self.cells_per_class.index.values]
        # Define pool of all possible class indices to choose from
        self.class_pool = list(np.arange(self.n_classes))
        # Sample from smallest classes first to ensure that they will be included in training
        if last_first:
            self.class_pool = self.class_pool[::-1]

    def _fill_batch(self, class_pool: list[int], idx_pools: list[set[int]]) -> tuple[list[int], list[int], list[int]]:
        # Set batch size
        batch_space = self.batch_size
        # Collect indices from random classes for a batch
        batch_idc = []
        # Keep drawing cells from classes until batch bin is full
        i = 0
        classes_in_batch = []
        cells_per_class = []
        # Randomly walk through the available classes
        if self.shuffle_classes:
            _class_pool = list(np.random.permutation(class_pool))
        
        while batch_space > 0 and len(_class_pool) > 0:
            target_class = _class_pool[i]
            batch_space = self.batch_size - len(batch_idc)
            # Get all available indices of that class
            target_pool = idx_pools[target_class]
            # Check number of available indices in class
            n_target_idc = len(target_pool)
            # Try to draw max_cells_per_class if we have that many
            if self.max_cells_per_batch <= n_target_idc:
                draw_n = self.max_cells_per_batch
            else:
                # Just draw as many as there are left in class
                draw_n = n_target_idc
                # Remove class from class pool
                _class_pool.remove(target_class)
                
            # Check if we can add enough cells to batch
            if draw_n > batch_space:
                draw_n = batch_space
            i += 1
            if draw_n > 0:
                # Randomly draw as many cells as we can from that class and add to batch
                target_idc = np.random.choice(list(target_pool), draw_n, replace=False)
                batch_idc.extend(target_idc)
                # Remove those indices from class pool
                target_pool -= set(target_idc)
                classes_in_batch.append(i)
                cells_per_class.append(draw_n)
            # Cycle through all classes repeatedly until either batch is full or classes are empty
            if i >= len(_class_pool):
                i = 0
        return batch_idc, classes_in_batch, cells_per_class

    def sample(self, copy: bool = False, return_details: bool = False) -> dict[str, np.ndarray | list]:
        batches = []
        n_classes_per_batch = []
        n_idc_per_batch = []
        classes_in_batches = []
        # Make copies of classes and indices
        if copy:
            class_pool = deepcopy(self.class_pool)
            idx_pools = deepcopy(self.idx_pools)
        else:
            class_pool = self.class_pool
            idx_pools = self.idx_pools
        # Draw cells for each batch
        for _ in np.arange(self.n_batches):
            batch_idc, classes_in_batch, cells_per_class = self._fill_batch(class_pool=class_pool, idx_pools=idx_pools)
            batches.extend(batch_idc)
            n_classes_per_batch.append(cells_per_class)
            n_idc_per_batch.append(len(batch_idc))
            classes_in_batches.append(classes_in_batch)
        # Count how many cells are left in each class after drawing
        class_pool_space = [len(c) for c in idx_pools]
        if return_details:
            return {
                self.IDX_KEY: np.array(batches),
                self.CLASSES_PER_BATCH_KEY: n_classes_per_batch,
                self.IDC_PER_BATCH_KEY: np.array(n_idc_per_batch),
                self.CLASS_POOL_INFO_KEY: np.array(class_pool_space)
            }
        else:
            return {
                self.IDX_KEY: np.array(batches)
            }
        

        