import pytest

from timeseries_zarr.constants import INNER_CHUNK_SAMPLES
from timeseries_zarr.sizing import (
    chunk_and_shard,
    chunk_shape_for_level,
    shard_shape_for_level,
)
from timeseries_zarr.types import ChunkShard


@pytest.mark.parametrize(
    ("level_shape", "inner_len", "expected"),
    [
        ((1000,), 256, (256,)),
        ((100,), 256, (100,)),
        ((256,), 256, (256,)),
        ((1000, 2), 256, (256, 2)),
        ((100, 2), 256, (100, 2)),
        ((0,), 256, (1,)),  # zero-length level -> one-bin floor
        ((0, 2), 256, (1, 2)),
    ],
)
def test_chunk_shape_for_level(level_shape, inner_len, expected):
    assert chunk_shape_for_level(level_shape, inner_len) == expected


def test_chunk_shape_for_level_default_inner_len_rank1():
    assert chunk_shape_for_level((10**9,)) == (INNER_CHUNK_SAMPLES,)


def test_chunk_shape_for_level_default_inner_len_rank2():
    assert chunk_shape_for_level((10**9, 2)) == (INNER_CHUNK_SAMPLES, 2)


@pytest.mark.parametrize(
    ("chunk_shape", "level_shape", "dtype_size", "target", "expected"),
    [
        (
            (262144,),
            (100_000_000,),
            4,
            16 * 2**20,
            (4194304,),
        ),  # target caps the shard
        ((256,), (100_000,), 4, 16 * 2**20, (100096,)),  # target exceeds array
        ((1024,), (100_000,), 4, 1024, (1024,)),  # chunk > target -> one chunk
        (
            (131072, 2),
            (100_000_000, 2),
            4,
            16 * 2**20,
            (2097152, 2),
        ),  # rank-2, 2-axis kept
        ((100,), (100,), 4, 16 * 2**20, (100,)),  # tiny -> one chunk
        ((1,), (0,), 4, 16 * 2**20, (1,)),  # zero-length level -> one chunk
    ],
)
def test_shard_shape_for_level(
    chunk_shape, level_shape, dtype_size, target, expected
):
    assert (
        shard_shape_for_level(chunk_shape, level_shape, dtype_size, target)
        == expected
    )


def test_shard_shape_for_level_default_target():
    assert shard_shape_for_level((262144,), (100_000_000,), 4) == (4194304,)


@pytest.mark.parametrize(
    ("chunk_shape", "level_shape", "dtype_size", "target"),
    [
        ((262144,), (100_000_000,), 4, 16 * 2**20),
        ((256,), (100_000,), 4, 16 * 2**20),
        ((131072, 2), (100_000_000, 2), 4, 16 * 2**20),
        ((1024,), (100_000,), 4, 1024),
    ],
)
def test_shard_shape_is_multiple_of_chunk(
    chunk_shape, level_shape, dtype_size, target
):
    shard = shard_shape_for_level(chunk_shape, level_shape, dtype_size, target)
    assert shard[0] % chunk_shape[0] == 0
    assert shard[1:] == chunk_shape[1:]


@pytest.mark.parametrize(
    ("level_shape", "dtype_size"),
    [
        ((100_000_000,), 4),
        ((100_000_000, 2), 4),
        ((1000,), 8),
        ((500, 2), 1),
    ],
)
def test_chunk_and_shard_composes(level_shape, dtype_size):
    result = chunk_and_shard(level_shape, dtype_size)
    chunk = chunk_shape_for_level(level_shape)
    shard = shard_shape_for_level(chunk, level_shape, dtype_size)
    assert result == ChunkShard(chunk_shape=chunk, shard_shape=shard)


def test_chunk_and_shard_custom_params():
    level_shape = (100_000_000,)
    result = chunk_and_shard(
        level_shape, 4, inner_len=1024, target_shard_bytes=64 * 1024
    )
    chunk = chunk_shape_for_level(level_shape, 1024)
    shard = shard_shape_for_level(chunk, level_shape, 4, 64 * 1024)
    assert result == ChunkShard(chunk_shape=chunk, shard_shape=shard)


def test_chunk_and_shard_concrete():
    result = chunk_and_shard((100_000_000,), 4)
    assert result.chunk_shape == (8192,)
    assert result.shard_shape == (4194304,)


@pytest.mark.parametrize(
    ("level_shape", "expected"),
    [
        ((0,), ChunkShard(chunk_shape=(1,), shard_shape=(1,))),
        ((0, 2), ChunkShard(chunk_shape=(1, 2), shard_shape=(1, 2))),
    ],
)
def test_chunk_and_shard_zero_length_level(level_shape, expected):
    assert chunk_and_shard(level_shape, 4) == expected


@pytest.mark.parametrize(
    ("level_shape", "dtype_size", "expected_shard"),
    [((100_000_000,), 4, (4194304,)), ((100_000_000, 2), 4, (2097152, 2))],
)
def test_chunk_and_shard_fills_the_shard_target(
    level_shape, dtype_size, expected_shard
):
    result = chunk_and_shard(level_shape, dtype_size)
    assert result.shard_shape == expected_shard
    assert result.chunk_shape[0] == INNER_CHUNK_SAMPLES
