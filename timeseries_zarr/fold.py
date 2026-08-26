"""Min/max pyramid folding: reduce a level to the next coarser one in blocks of 4.

Every fold uses plain min/max, so a NaN propagates into the bin holding it.
"""

import numpy as np
import numpy.typing as npt

from timeseries_zarr.constants import DECIMATION_FACTOR, ENVELOPE_PAIR_SIZE


def _block_split(
    length: int, block: int = DECIMATION_FACTOR
) -> tuple[int, int]:
    """Return (n_full, tail_len) for splitting length items into blocks.

    n_full is the count of complete blocks of size block; tail_len is the
    leftover count (0..block-1), kept as a final partial block with no padding.
    Raises ValueError if block is not positive.
    """
    if block <= 0:
        raise ValueError("block must be positive")
    return (length // block, length % block)


def fold_raw_block(raw: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold raw samples into (min, max) envelope pairs over disjoint blocks of 4.

    Takes rank-1 samples and returns rank-2 pairs. A final partial block of 1-3
    samples becomes one row.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(raw.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), ENVELOPE_PAIR_SIZE), dtype=np.float32
    )
    if n_full:
        full = raw[:split].reshape(n_full, block)
        out[:n_full, 0] = full.min(axis=1)
        out[:n_full, 1] = full.max(axis=1)
    if tail_len:
        tail = raw[split:]
        out[n_full, 0] = tail.min()
        out[n_full, 1] = tail.max()
    return out


def fold_pair_block(pairs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold (min, max) pairs into coarser pairs over disjoint blocks of 4.

    Each output row takes the smallest of the 4 mins (column 0) and the largest
    of the 4 maxes (column 1). A final partial block of 1-3 rows becomes one
    row.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(pairs.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), ENVELOPE_PAIR_SIZE), dtype=np.float32
    )
    if n_full:
        full = pairs[:split].reshape(n_full, block, ENVELOPE_PAIR_SIZE)
        out[:n_full, 0] = full[:, :, 0].min(axis=1)
        out[:n_full, 1] = full[:, :, 1].max(axis=1)
    if tail_len:
        tail = pairs[split:]
        out[n_full, 0] = tail[:, 0].min()
        out[n_full, 1] = tail[:, 1].max()
    return out


def fold_block(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold one pyramid level to the next coarser one, dispatching on rank.

    Rank-1 is raw samples, folded by fold_raw_block; rank-2 is (min, max)
    pairs, folded by fold_pair_block. Raises ValueError for any other rank.
    """
    match arr.ndim:
        case 1:
            return fold_raw_block(arr)
        case 2:
            return fold_pair_block(arr)
        case _:
            raise ValueError(f"unsupported array rank: {arr.ndim}")
