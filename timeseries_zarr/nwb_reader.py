"""Concrete NWB adapters implementing the channel-source protocols."""

from collections.abc import Iterator
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import Units

from timeseries_zarr.constants import (
    MAX_UNIT_CLUSTERS,
    MICROSECONDS_PER_SECOND,
    UNIT_TO_UV,
)


class NwbContinuousSource:
    """A continuous channel backed by one column of an NWB ElectricalSeries."""

    def __init__(
        self,
        electrical_series: ElectricalSeries,
        channel_index: int,
        session_start_time: datetime,
    ) -> None:
        """Bind one channel of an ElectricalSeries as a continuous source.

        channel_index selects the column this source exposes.
        session_start_time is the recording's wall-clock origin.
        """
        self._series = electrical_series
        self._channel_index = channel_index
        self._session_start_time = session_start_time

    @property
    def id(self) -> str:
        """The selected electrode's table id."""
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        return str(electrodes.table.id[row_index])

    @property
    def name(self) -> str:
        """The electrode's channel_name column, then its label, then the id."""
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        table = electrodes.table
        for column in ("channel_name", "label"):
            if column in table.colnames:
                return str(table[column][row_index])
        return self.id

    @property
    def unit(self) -> str:
        """Always "uV"; read_samples normalizes every series to microvolts."""
        return "uV"

    def rate_hz(self) -> float:
        """Return the ElectricalSeries' sample rate in hertz."""
        return float(self._series.rate)

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0.

        The session start plus the series' own start offset, rounded to whole
        microseconds.
        """
        start_s: float = (
            self._session_start_time.timestamp() + self._series.starting_time
        )
        return round(start_s * MICROSECONDS_PER_SECOND)

    def num_samples(self) -> int:
        """Return the length of the series' time axis, shared by every channel."""
        return int(self._series.data.shape[0])

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32 microvolts.

        Applies the series' affine scaling, then converts from the series' own
        unit to microvolts. Raises ValueError if that unit is not a recognized
        volts family. An empty range (stop <= start) yields a length-0 array.
        """
        column: npt.NDArray[np.float64] = np.asarray(
            self._series.data[start:stop, self._channel_index],
            dtype=np.float64,
        )
        scaled = column * float(self._series.conversion)
        if self._series.channel_conversion is not None:
            scaled = scaled * float(
                self._series.channel_conversion[self._channel_index]
            )
        scaled = scaled + float(self._series.offset)
        unit = str(self._series.unit).lower()
        if unit not in UNIT_TO_UV:
            raise ValueError(f"unsupported ElectricalSeries unit: {unit!r}")
        return (scaled * UNIT_TO_UV[unit]).astype(np.float32)


class NwbUnitSource:
    """A unit (spike) channel backed by an NWB Units table.

    Flattens the table's per-cluster rows into the per-event streams the
    bundle stores: all spikes merged into one timestamp series sorted
    ascending, each event tagged with its cluster's dense uint8 id in table
    row order (not the upstream unit id) and carrying that cluster's
    waveform_mean.
    """

    def __init__(
        self,
        units: Units,
        waveform_rate_hz: float,
        session_start_time: datetime,
    ) -> None:
        """Bind an NWB Units table as a flattened per-event spike source.

        waveform_rate_hz is the sample rate within a waveform; the table
        carries no rate of its own. session_start_time places event timestamps
        in absolute microseconds. Raises ValueError if the table holds more
        than 256 units, past the uint8 cluster-id range.
        """
        unit_count = len(units)
        if unit_count > MAX_UNIT_CLUSTERS:
            raise ValueError("a unit channel holds at most 256 clusters")

        self._id = str(units.name)
        self._rate_hz = float(waveform_rate_hz)
        self._start_us = round(
            session_start_time.timestamp() * MICROSECONDS_PER_SECOND
        )

        session_s = session_start_time.timestamp()
        times: list[npt.NDArray[np.float64]] = []
        clusters: list[npt.NDArray[np.uint8]] = []
        waveforms: list[npt.NDArray[np.float32]] = []
        for cluster_id in range(unit_count):
            spike_times: npt.NDArray[np.float64] = np.asarray(
                units.get_unit_spike_times(cluster_id), dtype=np.float64
            )
            mean: npt.NDArray[np.float32] = np.asarray(
                units["waveform_mean"][cluster_id], dtype=np.float32
            )
            count = spike_times.shape[0]
            times.append(spike_times)
            clusters.append(np.full(count, cluster_id, dtype=np.uint8))
            waveforms.append(np.broadcast_to(mean, (count, mean.shape[0])))

        all_times = np.concatenate(times)
        order = np.argsort(all_times, kind="stable")
        self._events = (
            ((all_times[order] + session_s) * MICROSECONDS_PER_SECOND)
            .round()
            .astype(np.int64)
        )
        self._units = np.concatenate(clusters)[order]
        self._waveforms = np.concatenate(waveforms)[order]

    @property
    def id(self) -> str:
        """The Units table's name."""
        return self._id

    @property
    def name(self) -> str:
        """The Units table's name, the same value as id."""
        return self._id

    @property
    def unit(self) -> str:
        """Always "uV".

        waveform_mean carries no unit metadata, so the amplitudes are stored
        unscaled.
        """
        return "uV"

    def rate_hz(self) -> float:
        """Return the waveform sample rate in hertz, as bound at construction."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the recording start, rounded to whole microseconds."""
        return self._start_us

    def num_events(self) -> int:
        """Return the total number of spike events across all units."""
        return int(self._events.shape[0])

    def points_per_event(self) -> int:
        """Return the width of the waveform_mean template, shared by every event."""
        return int(self._waveforms.shape[1])

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the half-open [start, stop) window of event timestamps.

        Absolute-microsecond int64. An empty range (stop <= start) yields a
        length-0 array.
        """
        return self._events[start:stop]

    def read_units(self, start: int, stop: int) -> npt.NDArray[np.uint8]:
        """Return the half-open [start, stop) window of per-event cluster ids.

        Aligned with the events at the same indices. An empty range yields a
        length-0 array.
        """
        return self._units[start:stop]

    def read_waveforms(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return float32 waveforms for events [start, stop).

        Row k is the waveform_mean of the cluster that produced event k. An
        empty range yields a (0, points_per_event) array.
        """
        return self._waveforms[start:stop]


def _iter_units_tables(nwbfile: NWBFile) -> Iterator[Units]:
    """Yield every Units table in the file in deterministic discovery order.

    The root Units table first when present, then the Units containers of each
    processing module, modules in name order and their containers in name
    order.
    """
    if nwbfile.units is not None:
        yield nwbfile.units
    for module_name in sorted(nwbfile.processing):
        module = nwbfile.processing[module_name]
        for container_name in sorted(module.data_interfaces):
            container = module.data_interfaces[container_name]
            if isinstance(container, Units):
                yield container


def build_sources_from_nwb(
    nwbfile: NWBFile,
) -> tuple[list[NwbContinuousSource], list[NwbUnitSource]]:
    """Discover the channel sources to write from an open NWB file.

    Every ElectricalSeries in the file's acquisition contributes one continuous
    source per channel column, in series order then channel order. Each Units
    table contributes one unit source, in _iter_units_tables order, with the
    waveform rate taken from the first ElectricalSeries. Raises ValueError if a
    Units table is present and no ElectricalSeries supplies that rate.
    """
    session_start = nwbfile.session_start_time
    series = [
        acq
        for acq in nwbfile.acquisition.values()
        if isinstance(acq, ElectricalSeries)
    ]
    continuous = [
        NwbContinuousSource(es, channel_index, session_start)
        for es in series
        for channel_index in range(es.data.shape[1])
    ]

    units: list[NwbUnitSource] = []
    for table in _iter_units_tables(nwbfile):
        if not series:
            raise ValueError("unit channels need an ElectricalSeries rate")
        waveform_rate_hz = float(series[0].rate)
        units.append(NwbUnitSource(table, waveform_rate_hz, session_start))

    return continuous, units
