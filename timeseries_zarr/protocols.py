"""Typed source protocols the writer consumes for channel data."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class ContinuousChannelSource(Protocol):
    """A continuous channel the writer reads raw samples from."""

    @property
    def id(self) -> str:
        """Opaque upstream channel identifier."""
        ...

    @property
    def name(self) -> str:
        """Human-readable display label for the channel."""
        ...

    @property
    def unit(self) -> str:
        """Physical unit of the stored samples, for example "uV"."""
        ...

    def rate_hz(self) -> float:
        """Return the channel's sample rate in hertz."""
        ...

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0."""
        ...

    def num_samples(self) -> int:
        """Return the total number of raw samples in the channel."""
        ...

    def offset_uv(self) -> float:
        """Return the channel's DC offset, in the unit its samples are stored in.

        The writer subtracts it before folding, so the statistics hold the
        signal without its bias and a reader adds it back in float64. A source
        that has no offset to declare returns 0.0.
        """
        ...

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32."""
        ...


@runtime_checkable
class UnitChannelSource(Protocol):
    """A unit (spike) channel the writer reads events, classifications, and waveforms from.

    rate_hz is the waveform sample rate, not an event rate.
    """

    @property
    def id(self) -> str:
        """Opaque upstream unit identifier."""
        ...

    @property
    def name(self) -> str:
        """Human-readable display label for the channel."""
        ...

    @property
    def unit(self) -> str:
        """Physical unit of the stored samples, for example "uV"."""
        ...

    def rate_hz(self) -> float:
        """Return the waveform sample rate in hertz."""
        ...

    def start_us(self) -> int:
        """Return the wall-clock microseconds of the recording start."""
        ...

    def num_events(self) -> int:
        """Return the total number of spike events."""
        ...

    def num_labels(self) -> int:
        """Return how many distinct labels the channel can use.

        The width of a count row, so it is the label space rather than the
        labels actually seen. 0 means the channel carries no labels.
        """
        ...

    def points_per_event(self) -> int:
        """Return the number of waveform samples stored per event."""
        ...

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of absolute-microsecond timestamps."""
        ...

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the half-open [start, stop) window of per-event labels.

        Cluster ids for sorted spikes. u2, so a high-density sort is not
        capped at 256.
        """
        ...

    def read_waveforms(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the float32 waveforms for events [start, stop).

        One row per event, points_per_event samples wide.
        """
        ...
