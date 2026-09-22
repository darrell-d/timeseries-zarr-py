"""Pyramid level planning: level counts, shapes, and time resolutions.

Raw samples are not a level. They are what the pyramid is built from, so they
carry no bin arithmetic and are written under their own key. Levels are
numbered from 1, which is what lets a reader treat every numeric-keyed child of
a channel as a level group without inspecting its shape.
"""

from timeseries_zarr.constants import (
    DECIMATION_FACTOR,
    ENVELOPE_PAIR_SIZE,
    MAX_LEVELS,
    MICROSECONDS_PER_SECOND,
)
from timeseries_zarr.types import LevelPlan


def level_count(
    num_samples: int, max_levels: int = MAX_LEVELS, min_bins: int = 1024
) -> int:
    """Return the number of level groups for a channel of num_samples samples.

    A level is included only while it would still hold at least min_bins
    complete bins, and the total is capped at max_levels. A partial trailing bin
    does not count toward min_bins, so this gate is one stricter than
    level_num_bins, which keeps the partial bin. A channel too short for level 1
    gets no levels at all, and neither does an empty one.
    """
    count = 0
    for level in range(1, max_levels + 1):
        if num_samples // DECIMATION_FACTOR**level < min_bins:
            break
        count += 1
    return count


def level_num_bins(num_samples: int, level: int) -> int:
    """Return the number of bins at a pyramid level for num_samples samples.

    Each level has one bin per disjoint block of 4 from the level below, and a
    partial final block becomes one bin. Level 1 folds the raw samples.
    """
    samples_per_bin: int = DECIMATION_FACTOR**level
    return -(-num_samples // samples_per_bin)


def sample_period_us(sample_rate_hz: float) -> float:
    """Return the spacing between raw samples in microseconds."""
    return MICROSECONDS_PER_SECOND / sample_rate_hz


def level_period_us(sample_period_us: float, level: int) -> float:
    """Return the microseconds that one bin spans at the given pyramid level.

    Each level spans 4x more time than the one below, and level 1 spans 4 raw
    samples.
    """
    return float(sample_period_us * DECIMATION_FACTOR**level)


def raw_shape(num_samples: int) -> tuple[int, ...]:
    """Return the raw array's shape: one axis, one entry per sample."""
    return (num_samples,)


def level_shape(num_samples: int, level: int) -> tuple[int, ...]:
    """Return the shape of a level's env member.

    One entry per bin, then a trailing axis of length 2 holding the (min, max)
    pair.
    """
    return (level_num_bins(num_samples, level), ENVELOPE_PAIR_SIZE)


def plan_levels(
    num_samples: int,
    sample_period_us: float,
    max_levels: int = MAX_LEVELS,
    min_bins: int = 1024,
) -> list[LevelPlan]:
    """Return the pyramid plan for a channel: one LevelPlan per level group.

    Numbered from 1. An empty list means the channel carries raw samples and no
    pyramid, which is what a channel shorter than one level's worth of bins
    gets.
    """
    return [
        LevelPlan(
            level=k,
            shape=level_shape(num_samples, k),
            period_us=level_period_us(sample_period_us, k),
        )
        for k in range(1, level_count(num_samples, max_levels, min_bins) + 1)
    ]
