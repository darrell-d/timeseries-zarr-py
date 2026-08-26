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

    def points_per_event(self) -> int:
        """Return the number of waveform samples stored per event."""
        ...

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of absolute-microsecond timestamps."""
        ...

    def read_units(self, start: int, stop: int) -> npt.NDArray[np.uint8]:
        """Return the half-open [start, stop) window of per-event cluster ids."""
        ...

    def read_waveforms(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the float32 waveforms for events [start, stop).

        One row per event, points_per_event samples wide.
        """
        ...
