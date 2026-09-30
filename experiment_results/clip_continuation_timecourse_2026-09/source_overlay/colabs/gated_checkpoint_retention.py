"""Hybrid rollout checkpoint layout; no changes to the training algorithm."""

import os
from pathlib import Path
import shutil
import tempfile

SEEDS = [789, 2026, 31415]
ARMS = ['baseline', 'always_on', 'step_gated', 'alignment_gated']
STEPS = [100, 250, 500, 750, 1001]
ROLLING = 'rolling_checkpoint.pt'


def permanent_path(seed, arm, step):
    if seed not in SEEDS or arm not in ARMS or step not in STEPS:
        raise ValueError('Unknown rollout checkpoint identity')
    if step == 100 and arm == 'baseline':
        return f'seed_{seed}/shared/checkpoints/step_000100.pt'
    if step == 1001 or (arm == 'alignment_gated' and step in [250, 500, 750]):
        return f'seed_{seed}/{arm}/checkpoints/step_{step:06d}.pt'
    return None


def required_paths():
    return sorted({name for seed in SEEDS for arm in ARMS for step in STEPS
                   if (name := permanent_path(seed, arm, step)) is not None})


def save_hybrid_checkpoint(root, state, *, seed, arm, step):
    """Publish one rolling file, plus selected immutable snapshots, atomically.

    The source tree contains no discarded historical checkpoints. DriveMirror
    verifies replacements before publishing them on Drive. Existing runs are
    never pruned or migrated by this code.
    """
    import torch
    root = Path(root)
    relative = permanent_path(seed, arm, step)
    if (state.get('training_seed'), state.get('arm'), state.get('completed_updates')) != (seed, arm, step):
        raise ValueError('State metadata does not match its retention identity')
    permanent = root / relative if relative else None
    if permanent is not None and permanent.exists():
        raise FileExistsError(f'Refusing to overwrite a permanent state: {permanent}')
    descriptor, name = tempfile.mkstemp(prefix='.rolling-', suffix='.pt', dir=root)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(state, temporary)
        temporary.replace(root / ROLLING)
        if permanent is not None:
            permanent.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix='.snapshot-', suffix='.pt', dir=permanent.parent)
            os.close(descriptor)
            snapshot = Path(name)
            try:
                shutil.copyfile(root / ROLLING, snapshot)
                snapshot.replace(permanent)
            finally:
                snapshot.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return relative
