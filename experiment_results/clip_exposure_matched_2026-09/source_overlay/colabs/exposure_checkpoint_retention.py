"""Separate immutable retention layout for the exposure-matched follow-up."""

import os
from pathlib import Path
import shutil
import tempfile

SEEDS = [789, 2026, 31415]
ARMS = ['alignment_gated', 'random_matched_1', 'random_matched_2', 'random_matched_3']
STEPS = [100, 250, 500, 750, 1001]
ROLLING = 'rolling_checkpoint.pt'


def permanent_path(seed, arm, step):
    if seed not in SEEDS or arm not in ARMS or step not in STEPS:
        raise ValueError('Unknown exposure-matched checkpoint identity')
    if step == 100 and arm == 'alignment_gated':
        return f'seed_{seed}/shared/checkpoints/step_000100.pt'
    if step == 1001 or (arm == 'alignment_gated' and step in [250, 500, 750]):
        return f'seed_{seed}/{arm}/checkpoints/step_{step:06d}.pt'
    return None


def required_paths():
    return sorted({p for seed in SEEDS for arm in ARMS for step in STEPS
                   if (p := permanent_path(seed, arm, step)) is not None})


def save_checkpoint(root, state, *, seed, arm, step):
    import torch
    root = Path(root)
    relative = permanent_path(seed, arm, step)
    if (state.get('training_seed'), state.get('arm'), state.get('completed_updates')) != (seed, arm, step):
        raise ValueError('Checkpoint metadata does not match retention identity')
    permanent = root / relative if relative else None
    if permanent is not None and permanent.exists():
        raise FileExistsError(f'Refusing permanent checkpoint overwrite: {permanent}')
    fd, name = tempfile.mkstemp(prefix='.rolling-', suffix='.pt', dir=root)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(state, temporary)
        temporary.replace(root / ROLLING)
        if permanent is not None:
            permanent.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.snapshot-', suffix='.pt', dir=permanent.parent)
            os.close(fd)
            snapshot = Path(name)
            try:
                shutil.copyfile(root / ROLLING, snapshot)
                snapshot.replace(permanent)
            finally:
                snapshot.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return relative


def verify_inventory(inventory):
    expected = set(required_paths()) | {ROLLING}
    if any(p not in inventory or inventory[p]['bytes'] <= 0 for p in expected):
        raise AssertionError('Incomplete exposure checkpoint inventory')
    if {p for p in inventory if p.endswith('.pt')} != expected:
        raise AssertionError('Unexpected checkpoint duplicates')
    return len(expected) - 1
