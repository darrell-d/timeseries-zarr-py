"""Format constants for the pyramid bundle."""

from typing import Final

DECIMATION_FACTOR: Final = 4
"""Samples folded into one bin per pyramid level (4x coarser each level)."""

MAX_LEVELS: Final = 8
"""Most pyramid levels a bundle can hold: level 0 plus levels 1 through 7."""

ENVELOPE_PAIR_SIZE: Final = 2
"""Length of the trailing (min, max) axis on every coarser pyramid level."""

MAX_UNIT_CLUSTERS: Final = 256
"""Most distinct clusters a unit channel can hold (the uint8 cluster-id range)."""

FLOAT32_BYTES: Final = 4
"""Byte width of one float32 sample, for shard-size computation."""

INT64_BYTES: Final = 8
"""Byte width of one int64 event timestamp, for shard-size computation."""

UINT8_BYTES: Final = 1
"""Byte width of one uint8 cluster id, for shard-size computation."""

MICROSECONDS_PER_SECOND: Final = 1_000_000.0
"""Microseconds in one second, for converting a sample rate to a period."""

INNER_CHUNK_SAMPLES: Final = 2**13
"""Target inner Zarr chunk length in samples (8192)."""

TARGET_SHARD_BYTES: Final = 16 * 2**20
"""Target outer shard size in bytes (~16 MiB) for sharded pyramid arrays."""

UNIT_TO_UV: Final = {
    "volts": 1e6,
    "v": 1e6,
    "millivolts": 1e3,
    "mv": 1e3,
    "microvolts": 1.0,
    "uv": 1.0,
}
"""Conversion factors from a volts-family unit name to microvolts."""
