import random
from collections import defaultdict

from torch.utils.data import Sampler


class StratifiedClassSampler(Sampler):
    """Yields batches of (classes_per_batch x images_per_class) indices.

    Every yielded batch contains exactly classes_per_batch distinct classes
    with images_per_class indices each. Classes with fewer than images_per_class
    examples are silently excluded.

    Call set_epoch(epoch) before each epoch so batches differ across epochs.
    """

    def __init__(self, class_labels, classes_per_batch, images_per_class, seed=42):
        self.classes_per_batch = classes_per_batch
        self.images_per_class = images_per_class
        self.seed = seed
        self._epoch = 0

        class_to_indices = defaultdict(list)
        for idx, label in enumerate(class_labels):
            class_to_indices[label].append(idx)

        self.class_buckets = {
            cls: idxs
            for cls, idxs in class_to_indices.items()
            if len(idxs) >= images_per_class
        }
        self.classes = sorted(self.class_buckets.keys())

        n = images_per_class
        k = classes_per_batch
        c = len(self.classes)
        chunks_per_class = min(len(v) // n for v in self.class_buckets.values())
        batches_per_round = c // k
        self._num_batches = chunks_per_class * batches_per_round

        print(
            f"StratifiedClassSampler: {c} classes, {chunks_per_class} chunks/class, "
            f"{batches_per_round} batches/round → {self._num_batches} batches/epoch "
            f"(batch = {k} classes × {n} images = {k * n} samples)"
        )

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        n = self.images_per_class
        k = self.classes_per_batch
        c = len(self.classes)

        # Shuffle class order and images within each class each epoch
        class_order = list(self.classes)
        rng.shuffle(class_order)

        # Build chunks: list of N-image index lists per class
        class_chunks = []
        for cls in class_order:
            idxs = list(self.class_buckets[cls])
            rng.shuffle(idxs)
            num_full = (len(idxs) // n) * n
            chunks = [idxs[i : i + n] for i in range(0, num_full, n)]
            class_chunks.append(chunks)

        num_rounds = min(len(ch) for ch in class_chunks)
        batches_per_round = c // k

        batches = []
        for r in range(num_rounds):
            round_chunks = [class_chunks[ci][r] for ci in range(c)]
            for b in range(batches_per_round):
                batch = []
                for ci in range(b * k, (b + 1) * k):
                    batch.extend(round_chunks[ci])
                rng.shuffle(batch)
                batches.append(batch)

        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return self._num_batches
