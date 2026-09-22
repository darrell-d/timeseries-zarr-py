"""Compose streaming and zarr I/O to write one continuous channel.

A channel is the raw samples plus a ladder of level groups above them. Raw is
written first and from the source; each level is folded from the one below,
level 1 from raw. Every writer shapes its array from the plan, takes its chunk
and shard shapes from sizing, then streams blocks in. An empty input creates the
array and writes nothing.
"""

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import channel_group_attrs, level_group_attrs
from timeseries_zarr.constants import DECIMATION_FACTOR, FLOAT32_BYTES
from timeseries_zarr.fold import fold_block
from timeseries_zarr.planning import plan_levels, raw_shape, sample_period_us
from timeseries_zarr.protocols import ContinuousChannelSource
from timeseries_zarr.sizing import chunk_and_shard
from timeseries_zarr.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_raw_blocks,
)
from timeseries_zarr.types import ChunkShard, LevelPlan, WriteOpts
from timeseries_zarr.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)

RAW_KEY = "raw"
"""Key of the full-resolution sample array within a channel group."""

ENV_KEY = "env"
"""Key of the (min, max) envelope member within a level group."""


def _write_blocks(
    array: ZarrArray, blocks: Iterable[npt.NDArray[np.float32]]
) -> None:
    """Write each block to array at its running axis-0 offset.

    Blocks arrive shard-sized, so each write covers a whole shard. A write
    narrower than a shard makes the sharding codec read that shard back,
    re-encode every inner chunk, and rewrite it.
    """
    start = 0
    for block in blocks:
        write_region(array, start, block)
        start += block.shape[0]


def write_raw(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the raw array under group and stream the source's samples into it.

    Rank-1 float32 under the key "raw", with no attributes: the sample period is
    the channel's rate_hz and restating it here would be a second place to keep
    it right. The source is read in shard-sized blocks.
    """
    array = create_array(
        group,
        RAW_KEY,
        raw_shape(source.num_samples()),
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
    )
    _write_blocks(array, iter_raw_blocks(source, sizing.shard_shape[0]))
    return array


def write_level_from_previous(
    parent: ZarrGroup,
    prev: ZarrArray,
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the level group named plan.level and fold prev into its env member.

    The group carries period_us and holds one array per statistic over a shared
    bin axis; env is the only one so far. prev is the array below, raw for
    level 1 and the level below's env after that, read in axis-0 blocks of
    DECIMATION_FACTOR shards and folded across block boundaries by fold_block,
    so each folded block fills one shard. Returns the env array, which the next
    level folds from. Raises ValueError if plan describes a level below 1.
    """
    if plan.level < 1:
        raise ValueError("levels are numbered from 1; raw is not a level")

    group = create_group_with_attrs(
        parent, str(plan.level), level_group_attrs(plan.period_us)
    )
    array = create_array(
        group,
        ENV_KEY,
        plan.shape,
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        {},
        zstd_level,
    )

    _write_blocks(
        array,
        _rebuffer_and_fold(
            iter_array_blocks(
                cast("BlockReadableArray", prev),
                DECIMATION_FACTOR * sizing.shard_shape[0],
            ),
            fold_block,
        ),
    )
    return array


def write_continuous_channel(
    parent: ZarrGroup,
    index: int,
    source: ContinuousChannelSource,
    *,
    onset_us: int,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(index).

    Creates the channel group under parent carrying its continuous-kind
    attributes, writes the raw samples, then folds each level from the one
    written below it. The pyramid is planned from the source's sample count and
    rate; every array is sized and compressed per opts. A channel too short to
    fill one level gets raw and nothing else.

    onset_us is the bundle's onset in wall-clock microseconds. The channel
    records its distance from it, not its own wall-clock start: no absolute
    time may appear outside meta/.
    """
    attributes = channel_group_attrs(
        source.id,
        source.rate_hz(),
        source.start_us() - onset_us,
        "continuous",
        source.name,
        source.unit,
    )
    group = create_group_with_attrs(parent, str(index), attributes)

    def _sizing(shape: tuple[int, ...]) -> ChunkShard:
        return chunk_and_shard(
            level_shape=shape,
            dtype_size=FLOAT32_BYTES,
            inner_len=opts.inner_len,
            target_shard_bytes=opts.target_shard_bytes,
        )

    num_samples = source.num_samples()
    previous = write_raw(
        group=group,
        source=source,
        sizing=_sizing(raw_shape(num_samples)),
        zstd_level=opts.zstd_level,
    )
    for plan in plan_levels(
        num_samples,
        sample_period_us(source.rate_hz()),
        opts.max_levels,
        opts.min_bins,
    ):
        previous = write_level_from_previous(
            parent=group,
            prev=previous,
            plan=plan,
            sizing=_sizing(plan.shape),
            zstd_level=opts.zstd_level,
        )
