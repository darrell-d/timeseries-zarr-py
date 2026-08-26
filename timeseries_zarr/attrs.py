"""Build the attribute dicts written into the bundle's zarr.json files.

docs/bundle-format.md specifies the keys and their types.
"""

from timeseries_zarr.types import ChannelKind


def root_attrs() -> dict[str, object]:
    """Return the custom attributes for the bundle's root group.

    Always empty. The root carries only Zarr's own consolidated_metadata.
    """
    return {}


def channel_group_attrs(
    id: str,
    rate_hz: float,
    start_us: int,
    kind: ChannelKind,
    name: str = "",
    unit: str = "",
) -> dict[str, object]:
    """Return the channel-group zarr.json attributes for one channel.

    Values pass through unchanged.
    """
    return {
        "id": id,
        "rate_hz": rate_hz,
        "start_us": start_us,
        "kind": kind,
        "name": name,
        "unit": unit,
    }


def level_array_attrs(period_us: float) -> dict[str, object]:
    """Return the pyramid-level array attributes.

    period_us is the microseconds one bin spans at this level.
    """
    return {"period_us": period_us}


def waveform_array_attrs(period_us: float) -> dict[str, object]:
    """Return the waveforms array attributes for a unit channel.

    period_us is the sample period within a spike waveform, not a
    pyramid-level period.
    """
    return {"period_us": period_us}
