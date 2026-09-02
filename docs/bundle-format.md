# Viewer bundle format

A viewer bundle is a static directory of Zarr v3 arrays holding one recording at several
time resolutions. A browser opens it over HTTP range requests and renders any time
window without downloading the recording. This document specifies the format so that
other producers can write it and other readers can consume it.

`timeseries-zarr-py` is the reference producer. `@pennsieve/timeseries-zarr-reader` is the
reference reader.

## Why a pyramid

A recording at 32 kHz across 24 channels for 24 hours is about 265 GB in float32. A
viewer draws an overview into roughly 2000 pixels. Fetching raw samples to compute those
2000 columns is intractable in a browser, so the producer precomputes min/max envelopes at
several resolutions and the reader fetches only the bins one viewport needs.

The container is stock Zarr v3. There is no sidecar manifest, no schema language, and no
namespace prefix. The only custom surface is seven attribute keys within the standard Zarr
`attributes` field: six on each channel group and one on each level and waveform array.
Any Zarr v3 library can open a bundle.

## Layout

```
<bundle-root>/
  zarr.json              # root group; consolidated_metadata inlines every descendant
  0/                     # continuous channel
    zarr.json            # attributes: id, rate_hz, start_us, kind, name, unit
    0/                   # pyramid level 0: raw samples
      zarr.json          # shape (N,), dtype f4, attributes: period_us
      c/...              # shard files
    1/                   # pyramid level 1: 4x decimated min/max
      zarr.json          # shape (ceil(N/4), 2), dtype f4, attributes: period_us
      c/...
    2/                   # 16x
    7/                   # 16384x, the coarsest level the format allows
  1/                     # continuous channel
  23/                    # unit (spike) channel
    zarr.json            # attributes: id, rate_hz, start_us, kind, name, unit
    events/              # shape (n_events,), dtype i8
    units/               # shape (n_events,), dtype u1
    waveforms/           # shape (n_events, points_per_event), dtype f4
```

Channel groups are named by index (`0/`, `1/`, and so on). The index is opaque and its
order is arbitrary; upstream identifiers never appear in a path. The reader builds an
`id` to index map from consolidated metadata.

One group per channel, rather than one `(channel, time)` array, lets channels differ in
sample rate. A recording often mixes 32 kHz intracranial EEG with 1 kHz scalp EEG and
250 Hz pulse oximetry, and a shared array would force a common rate or padding. Separate
groups also let a producer write one channel at a time and let a reader fetch only the
channels on screen.

## Continuous channels

A continuous channel is a pyramid of arrays, one per level, keyed by level number.

Level 0 holds the raw signal: shape `(N,)`, dtype `float32`. Each level `k` at or above 1
holds interleaved `(min, max)` pairs: shape `(ceil(N / 4^k), 2)`, dtype `float32`. Column
0 holds each bin's min and column 1 its max. One array per level means the reader gets
both envelopes in a single fetch. Paired `min`/`max` arrays would double the request
count for no gain.

The fold is exact and needs one chunk of memory at a time:

- Level 1 takes the min and the max over each disjoint block of 4 raw samples.
- Level `k+1` takes the min of the 4 mins and the max of the 4 maxes over each disjoint
  block of 4 pairs from level `k`.

A trailing partial block of 1 to 3 elements produces a final bin.

The fold uses plain min and max and not its NaN-aware variants. A NaN therefore propagates
up the pyramid, and the reader must treat NaN as a gap.

A bundle holds at most 8 levels: level 0 plus levels 1 through 7, a range of 16384x. The
producer stops early once the next level runs out of complete bins: level `k` at or above
1 is present when `floor(N / 4^k)` is at least the bin threshold, and the written level
still keeps its partial trailing bin. The threshold is producer-tunable; the reference
producer defaults to 1024. A channel with no samples writes a zero-length level 0 and no
coarser levels.

The pyramid costs about 1.67x the raw size. Each level above raw stores a pair, so the
series `N + 2 * (N/4 + N/16 + ...)` sums to `5N/3`.

Samples are stored in microvolts. Sample index `i` at level `L` starts at wall-clock
`start_us + i * period_us(L)`. No time axis is stored.

## Unit channels

A unit channel holds spike events in three arrays of matching length. Index `k` refers to
the same event in all three.

| Array | Shape | dtype | Contents |
|---|---|---|---|
| `events` | `(n_events,)` | `i8` | absolute timestamp in microseconds, non-decreasing |
| `units` | `(n_events,)` | `u1` | cluster id, so at most 256 clusters per channel |
| `waveforms` | `(n_events, points_per_event)` | `f4` | waveform samples around the event |

Event timestamps are absolute, unlike continuous samples, which are indexed off the
channel's `start_us`. Equal timestamps are allowed: two clusters can fire in the same
microsecond.

Unit channels have no pyramid. Spikes are sparse next to a continuous signal, so the
volume is already bounded. A reader binary-searches the sorted `events` array to find
a window, which touches one or two chunks. A unit channel with no events writes all
three arrays at length 0.

## Attributes

Attribute keys are unprefixed and case sensitive. Every value lives in the standard Zarr
`attributes` blob.

The root group carries only Zarr's own `consolidated_metadata`.

Each channel group carries six keys:

| Key | Type | Meaning |
|---|---|---|
| `id` | string | upstream channel identifier the reader joins on for platform metadata |
| `rate_hz` | float64 | sample rate; for a unit channel, the waveform sample rate |
| `start_us` | int64 | wall-clock microseconds of sample 0, or of the recording start |
| `kind` | string | `continuous` or `unit` |
| `name` | string | display label |
| `unit` | string | physical unit of the stored samples, always `uV` |

Each pyramid level array and each `waveforms` array carries one key:

| Key | Type | Meaning |
|---|---|---|
| `period_us` | float64 | microseconds one bin spans, or one waveform sample for `waveforms` |

`period_us` is what drives the reader's level selection. `events` and `units` carry no
custom attributes; their name, shape, and dtype define them.

A level's layout needs no attribute. Rank 1 means raw samples and rank 2 with a trailing
dimension of 2 means `(min, max)` pairs.

## Storage

Arrays are Zstd-compressed and sharded with the ZEP2 sharding codec. The inner chunk
spans up to 2^13 (8192) bins along the time axis. The outer shard groups whole inner
chunks up to about 16 MiB; a single chunk already over that target forms a shard on its
own. The length-2 envelope axis is never chunked. The unit-channel arrays follow the
same sizing, and the `points_per_event` axis of `waveforms` is never chunked. The
compression level is a producer tuning knob, not part of the format.

The inner chunk is the smallest unit a reader can fetch, so its width sets the floor on
what a read transfers. A reader picks the level whose `period_us` matches one pixel,
which puts a rendered window at a few thousand bins whatever the zoom. A chunk wider
than that transfers bins nothing reads.

Sharding gives one file per channel per level instead of thousands of chunk files, and
the reader pulls byte ranges out of it. Reading a shard starts with reading its index
from the end of the file. A store should ask for a suffix range instead of issuing a HEAD
to learn the object size followed by an absolute-offset GET.

Every shard of every level is present on disk. A shard whose bins all equal the fill
value is still written, so a reader never meets a missing key. An object store that
answers a missing key with an authorization error instead of a not-found status would
otherwise turn an all-zero channel into a failed read. A reader that does meet a missing
shard reads the fill value, which is indistinguishable from recorded zeros.

`float32` is the dtype throughout. A viewer quantizes to canvas pixels, so the extra
precision of `float64` reaches no screen and costs twice the storage.

## Consolidated metadata

The root `zarr.json` must carry a Zarr v3 `consolidated_metadata` block inlining every
descendant `zarr.json`. This allows a single fetch to produce the whole trees
metadata of  channels, levels, shapes, dtypes, and attributes.

Otherwise a bundle of N channels costs about `8N + 1` metadata requests before the
first chunk fetch.

## Compatibility

There is no `format_version` attribute. Zarr's own mechanisms carry forward
compatibility: new attribute keys are additive and readers ignore what they do not
recognize, and new arrays are discovered through consolidated metadata enumeration rather
than assumed.
