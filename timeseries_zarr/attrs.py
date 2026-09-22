"""Build the attribute dicts written into the bundle's zarr.json files.

The format spec specifies the keys and their types:
https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md
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
    offset_us: int,
    kind: ChannelKind,
    name: str = "",
    unit: str = "",
) -> dict[str, object]:
    """Return the channel-group zarr.json attributes for one channel.

    offset_us is microseconds from the bundle's onset to this channel's
    first sample or event, never a wall-clock value. Values pass through
    unchanged.
    """
    return {
        "id": id,
        "rate_hz": rate_hz,
        "offset_us": offset_us,
        "kind": kind,
        "name": name,
        "unit": unit,
    }


def meta_group_attrs(
    subject: dict[str, object],
    session: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    """Return the meta/ group attributes: the bundle's whole identity surface.

    The three objects are deliberately unschematized; ontology belongs to
    archival standards. session carries start_us, the only wall-clock value
    anywhere in a bundle, which is what makes date-shifting a one-field edit.
    """
    return {"subject": subject, "session": session, "source": source}


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
