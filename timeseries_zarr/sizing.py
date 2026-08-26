"""Chunk and shard shape computation for Zarr pyramid arrays."""

from math import ceil, prod

from timeseries_zarr.constants import INNER_CHUNK_SAMPLES, TARGET_SHARD_BYTES
from timeseries_zarr.types import ChunkShard


def chunk_shape_for_level(
    level_shape: tuple[int, ...], inner_len: int = INNER_CHUNK_SAMPLES
) -> tuple[int, ...]:
    """Return the inner Zarr chunk shape for a pyramid level.

    The chunk spans up to inner_len bins along the time axis (axis 0), never
    more than the level's own length and never fewer than one bin, so a
    zero-length level still gets a shape Zarr accepts. The trailing length-2
    (min, max) axis of an envelope level is never chunked.
    """
    n, *rest = level_shape
    return (max(1, min(n, inner_len)), *rest)


def shard_shape_for_level(
    chunk_shape: tuple[int, ...],
    level_shape: tuple[int, ...],
    dtype_size: int,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> tuple[int, ...]:
    """Return the outer Zarr shard shape for a pyramid level.

    The shard spans as many whole inner chunks along the time axis (axis 0) as
    fit in target_shard_bytes, at least one and at most the number of chunks
    the level holds. A zero-length level takes one chunk. It is always an
    integer multiple of chunk_shape along axis 0; the trailing length-2
    (min, max) axis stays size 2.
    """
    chunk_bytes = prod(chunk_shape) * dtype_size
    chunks_per_shard = max(1, target_shard_bytes // chunk_bytes)
    chunks_in_level = ceil(level_shape[0] / chunk_shape[0])
    shard_chunks = max(1, min(chunks_per_shard, chunks_in_level))
    return (shard_chunks * chunk_shape[0], *chunk_shape[1:])


def chunk_and_shard(
    level_shape: tuple[int, ...],
    dtype_size: int,
    inner_len: int = INNER_CHUNK_SAMPLES,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> ChunkShard:
    """Return the inner-chunk and outer-shard shapes for a level as a ChunkShard."""
    chunk_shape = chunk_shape_for_level(level_shape, inner_len)
    shard_shape = shard_shape_for_level(
        chunk_shape, level_shape, dtype_size, target_shard_bytes
    )

    return ChunkShard(chunk_shape=chunk_shape, shard_shape=shard_shape)
