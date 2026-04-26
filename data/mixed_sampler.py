import random
from collections import defaultdict

from torch.utils.data import Sampler


class MixedBatchSampler(Sampler):
    """Yields batches mixing a stratified within-class block with a random block.

    Each batch contains:
      - classes_per_batch × images_per_class stratified indices (within-class hard negatives)
      - batch_size - (classes_per_batch × images_per_class) random indices (cross-class diversity)

    Default at batch_size=64, classes_per_batch=4, images_per_class=4:
      16 stratified (25%) + 48 random (75%) per batch. All 64 indices are distinct.

    Call set_epoch(epoch) before each epoch so batches differ across epochs.
    """

    def __init__(self, class_labels, batch_size, classes_per_batch=4, images_per_class=4, seed=42):
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.images_per_class = images_per_class
        self.seed = seed
        self._epoch = 0

        self.all_indices = list(range(len(class_labels)))
        self.stratified_count = classes_per_batch * images_per_class
        self.random_count = batch_size - self.stratified_count

        assert self.random_count > 0, (
            f"random_count={self.random_count}: classes_per_batch × images_per_class "
            f"must be less than batch_size"
        )

        class_to_indices = defaultdict(list)
        for idx, label in enumerate(class_labels):
            class_to_indices[label].append(idx)

        self.class_buckets = {
            cls: idxs
            for cls, idxs in class_to_indices.items()
            if len(idxs) >= images_per_class
        }
        self.classes = sorted(self.class_buckets.keys())

        if len(self.classes) < classes_per_batch:
            raise ValueError(
                f"Not enough eligible classes: found {len(self.classes)}, "
                f"need {classes_per_batch}. Reduce classes_per_batch or images_per_class."
            )

        # Same number of batches as a pure random loader with drop_last=True
        self._num_batches = len(self.all_indices) // batch_size

        pct = round(100 * self.stratified_count / batch_size)
        print(
            f"MixedBatchSampler: {len(self.classes)} eligible classes, "
            f"{self.stratified_count} stratified + {self.random_count} random per batch "
            f"({pct}% within-class), {self._num_batches} batches/epoch"
        )

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        random_pool = list(self.all_indices)
        rng.shuffle(random_pool)
        pool_cursor = 0

        for _ in range(self._num_batches):
            # Stratified block: pick classes_per_batch distinct classes, images_per_class each
            chosen_classes = rng.sample(self.classes, self.classes_per_batch)
            strat_indices = []
            for cls in chosen_classes:
                strat_indices.extend(rng.sample(self.class_buckets[cls], self.images_per_class))
            strat_set = set(strat_indices)

            # Random block: consume from pool, skipping any index already in stratified block
            rand_indices = []
            while len(rand_indices) < self.random_count:
                if pool_cursor >= len(random_pool):
                    pool_cursor = 0
                    rng.shuffle(random_pool)
                idx = random_pool[pool_cursor]
                pool_cursor += 1
                if idx not in strat_set:
                    rand_indices.append(idx)

            batch = strat_indices + rand_indices
            rng.shuffle(batch)
            yield batch
