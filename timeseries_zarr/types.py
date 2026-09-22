"""Shared value types for the writer."""

from dataclasses import dataclass
from typing import Literal

from timeseries_zarr.constants import (
    INNER_CHUNK_SAMPLES,
    MAX_LEVELS,
    TARGET_SHARD_BYTES,
)

type ChannelKind = Literal["continuous", "unit"]


@dataclass(frozen=True, slots=True)
class RecordingMeta:
    """Recording-level metadata for the bundle's meta/ group.

    Everything identifying a bundle carries lives in these three objects and
    nowhere else, so that deleting one directory de-identifies it. session
    gains start_us when the bundle is written; the onset is the earliest
    start across every channel, which only the bundle knows.
    """

    subject: dict[str, object]
    session: dict[str, object]
    source: dict[str, object]


@dataclass(frozen=True, slots=True)
class LevelPlan:
    """Resolved shape and time resolution of one pyramid level of a channel.

    Level 0 holds raw samples; levels >= 1 hold (min, max) envelopes. period_us
    is the wall-clock microseconds that one bin spans, widening with each
    coarser level.
    """

    level: int
    shape: tuple[int, ...]
    period_us: float

    @property
    def is_raw(self) -> bool:
        """Whether this level holds raw samples rather than min/max envelopes."""
        return self.level == 0

    @property
    def name(self) -> str:
        """Zarr array key for this level: its decimal level number."""
        return str(self.level)


@dataclass(frozen=True, slots=True)
class ChunkShard:
    """Inner-chunk and outer-shard shapes for one Zarr array.

    chunk_shape is the inner (compressed) chunk; shard_shape is the outer shard
    that groups whole chunks under the ZEP2 sharding codec.
    """

    chunk_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WriteOpts:
    """Tunable settings shared by the channel writers.

    max_levels and min_bins bound the pyramid; inner_len and
    target_shard_bytes size the inner chunk and the outer shard.
    """

    zstd_level: int = 5
    max_levels: int = MAX_LEVELS
    min_bins: int = 1024
    inner_len: int = INNER_CHUNK_SAMPLES
    target_shard_bytes: int = TARGET_SHARD_BYTES
