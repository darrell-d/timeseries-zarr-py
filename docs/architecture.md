# Architecture

Data flows one way through the writer: read, plan, stream and fold, write, publish. No
stage reaches back into an earlier one.

## Input adapter

`protocols.py` declares `ContinuousChannelSource` and `UnitChannelSource`. Each exposes
one channel's metadata and windowed reads (`read_samples`, `read_events`, and the rest).

NWB is the only input format so far. Everything downstream depends on the protocols, not
on NWB. The core is testable against in-memory sources, and a second input format costs
one new adapter and no changes elsewhere.

`nwb_series.py` holds the readers every adapter shares: the rate and start time of a
series, one channel's window as float64 or as microvolts, an electrode's id and display
name. Keeping them here lets each adapter live in its own module without importing
another.

`nwb_reader.py` holds the rate-sampled adapters and the discovery that picks them.
`NwbContinuousSource` reads one column of an `ElectricalSeries` and normalizes it to
microvolts. `NwbTimeSeriesSource` reads one column of any other numeric `TimeSeries` in
the acquisition, normalized to microvolts when its unit is in the volts family and kept
in its own unit otherwise. Unit-channel waveforms carry no unit metadata in NWB and are
stored unscaled. `build_sources_from_nwb` chooses an adapter per series: a rate picks
these, timestamps pick the one below.

`nwb_timestamped.py` holds `NwbTimestampedSource`, which presents a recording with breaks
in it as a uniform grid. A series sampled by timestamps rather than a rate is how NWB has
to express one, since a rate asserts unbroken regularity. The adapter reports the grid's
length as its sample count and NaN wherever nothing was recorded, so nothing downstream
needs gap awareness. See [gapped recordings](./gapped-recordings.md).

## Decision layer

`planning.py`, `sizing.py` and `grid.py` are pure functions with no I/O. `planning.py`
decides how many levels a channel gets and each level's shape and `period_us`.
`sizing.py` decides the inner chunk and outer shard shapes for one array. `grid.py` maps
a recording's timestamps onto the uniform grid they occupy, returning one row per
contiguous run rather than per sample, and measures the rate when none is supplied; it
imports neither NWB nor Zarr.

Keeping these separate from the write path means the format's arithmetic is unit-testable
without touching a store.

## Numeric core

`fold.py` reduces one level to the next: min and max over disjoint blocks of 4. See
[the format spec](https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md) for the exact rule and the NaN behavior.

`streaming.py` drives the fold over a source one block at a time and buffers across block
boundaries so a run of 4 that straddles two blocks still folds correctly. Memory stays
bounded no matter how long the recording is.

## Write path

`write_continuous.py` and `write_unit.py` each write one channel by composing the stages
above. They hold the per-channel logic and no Zarr specifics.

Both pick their read block so that every write covers a whole shard. A narrower write
makes the sharding codec read the shard back, re-encode every inner chunk, and rewrite
it, which costs about 10x the store traffic on a 16-chunk shard.

`zarr_io.py` is the only module that imports `zarr`. Everything else is Zarr-agnostic, so
the Zarr v3 API surface this package depends on sits in one file.

`attrs.py` builds the attribute dicts, the format's only custom surface.

## Orchestration

`bundle.py` runs the whole job: assign channel indices, write each channel, consolidate
metadata, publish atomically.

`main.py` and `config.py` are the CLI and environment-config shell around it.
`config.py` resolves both invocation forms, positional arguments and the
`INPUT_DIR`/`OUTPUT_DIR` convention. After publishing, `properties.py` writes the
`asset-properties.json` sidecar that records the bundle's directory name.
