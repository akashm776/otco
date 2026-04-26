import random
from collections import defaultdict

from torch.utils.data import Sampler


class MixedBatchSampler(Sampler):
    """Yields batches mixing a stratified within-class block with a random block.

    Each batch contains:
      - classes_per_batch × images_per_class stratified indices (within-class hard negatives)
      - batch_size - (classes_per_batch × images_per_class) random indices (cross-class diversity)

    Default at batch_size=64, classes_per_batch=4, images_per_class=4:
      16 stratified (25%) + 48 random (75%) per batch.

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

        # Pre-build the random pool: shuffle all indices, consume random_count per batch
        random_pool = list(self.all_indices)
        rng.shuffle(random_pool)
        # random_pool has len(all_indices) elements; we need _num_batches * random_count
        # which is always <= len(all_indices) since random_count < batch_size

        for i in range(self._num_batches):
            # Stratified block: pick classes_per_batch distinct classes, images_per_class each
            chosen_classes = rng.sample(self.classes, self.classes_per_batch)
            strat_indices = []
            for cls in chosen_classes:
                pool = self.class_buckets[cls]
                strat_indices.extend(rng.sample(pool, self.images_per_class))

            # Random block: sequential slice through pre-shuffled pool
            rand_start = i * self.random_count
            rand_indices = random_pool[rand_start : rand_start + self.random_count]

            batch = strat_indices + rand_indices
            rng.shuffle(batch)
            yield batch
