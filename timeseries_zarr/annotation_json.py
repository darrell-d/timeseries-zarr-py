"""Read the annotation interchange document the extractors produce.

Every annotation source -- baked into an EDF, a sidecar, a detector run, a
database -- is normalised by its own extractor into one JSON document, and the
writer takes NWB plus those documents as its inputs. Annotations do not travel
inside NWB because annotations from a database have no NWB home at all, so an
NWB-only path would have needed a second path anyway.

One document is one annotation source and becomes one event channel. A
recording with EDF-baked marks, a detector run and a clinical export produces
three documents and three channels.

## The timebase

`time_reference: "recording_onset"` means `time_us` is microseconds from the
recording's onset, and the onset is the NWB file's `session_start_time`. The
extractor does the conversion; this reader only rebases onto the absolute
timeline the rest of the writer works in, and the bundle rebases everything once
more onto its own onset.

That chain is the highest-risk thing here. Nothing errors if it is wrong: the
annotations simply sit at the wrong time, and at 2 kHz nobody sees it by eye.
The one assumption worth stating is the one above -- recording onset is the NWB
session start -- so a document whose extractor meant something else by it will
be silently off by the difference.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SCHEMA_PREFIX = "pennsieve/annotations-"
"""Documents this reader understands. The document carries its own version."""

SUPPORTED_SCHEMAS = frozenset({"pennsieve/annotations-1.0"})

RECORDING_ONSET = "recording_onset"
"""The only time reference defined so far."""

JSON_MEDIA_TYPE = "application/json"


class AnnotationDocumentError(ValueError):
    """The document is not one this writer can turn into a channel."""


def _require(document: dict[str, Any], key: str, where: str) -> object:
    """Return a required field, naming what was missing rather than KeyError."""
    if key not in document:
        raise AnnotationDocumentError(f"{where} is missing {key!r}")
    return document[key]


class JsonAnnotationSource:
    """One annotation document, presented as an annotation channel source.

    The columns a channel carries follow from the document: durations appear if
    any mark has one, labels if any mark is labelled, values if any carries a
    measurement, bodies if the channel declares a media type.
    """

    def __init__(
        self,
        document: dict[str, Any],
        recording_onset_us: int,
        *,
        channel_index_by_name: dict[str, int] | None = None,
    ) -> None:
        """Bind one parsed document as a channel source.

        recording_onset_us is the wall-clock microsecond the document's times
        are measured from. channel_index_by_name resolves montage names to the
        bundle's channel indices; without it a document naming channels is
        rejected rather than written with references that point nowhere.
        """
        schema = _require(document, "schema", "document")
        if schema not in SUPPORTED_SCHEMAS:
            raise AnnotationDocumentError(
                f"unsupported schema {schema!r}: this writer reads "
                f"{sorted(SUPPORTED_SCHEMAS)}"
            )

        channel: dict[str, Any] = _require(  # type: ignore[assignment]
            document, "channel", "document"
        )
        self._id = str(_require(channel, "id", "channel"))
        self._name = str(channel.get("name", self._id))
        self._unit = str(channel.get("unit", ""))
        media_type = channel.get("body_media_type")
        self._media_type: str | None = (
            None if media_type is None else str(media_type)
        )

        reference = channel.get("time_reference", RECORDING_ONSET)
        if reference != RECORDING_ONSET:
            raise AnnotationDocumentError(
                f"unsupported time_reference {reference!r}: only "
                f"{RECORDING_ONSET!r} is defined"
            )

        marks: list[dict[str, Any]] = list(
            _require(document, "annotations", "document")  # type: ignore[call-overload]
        )
        marks.sort(key=lambda mark: int(mark["time_us"]))

        self._events = np.array(
            [recording_onset_us + int(mark["time_us"]) for mark in marks],
            dtype=np.int64,
        )
        self._durations = (
            np.array(
                [int(mark.get("duration_us", 0)) for mark in marks],
                dtype=np.int64,
            )
            if any("duration_us" in mark for mark in marks)
            else None
        )
        self._values = (
            np.array(
                [float(mark.get("value", np.nan)) for mark in marks],
                dtype=np.float32,
            )
            if any("value" in mark for mark in marks)
            else None
        )

        # Label names are the distinct labels in the order they first appear,
        # so the mapping is stable for one document and needs nothing stored.
        names: list[str] = []
        seen: dict[str, int] = {}
        for mark in marks:
            label = mark.get("label")
            if label is not None and label not in seen:
                seen[label] = len(names)
                names.append(str(label))
        self._label_names = names or None
        self._labels = (
            np.array(
                [seen.get(str(mark.get("label")), 0) for mark in marks],
                dtype=np.uint16,
            )
            if names
            else None
        )

        self._bodies = (
            [self._encode_body(mark) for mark in marks]
            if self._media_type is not None
            else None
        )
        self._refs = self._resolve_refs(marks, channel_index_by_name)

    def _encode_body(self, mark: dict[str, Any]) -> bytes:
        """Return one mark's payload as bytes, in the channel's media type."""
        if self._media_type == JSON_MEDIA_TYPE:
            body = mark.get("body", mark.get("text", {}))
            return json.dumps(body, separators=(",", ":")).encode()
        return str(mark.get("text", mark.get("body", ""))).encode()

    def _resolve_refs(
        self,
        marks: list[dict[str, Any]],
        index_by_name: dict[str, int] | None,
    ) -> list[npt.NDArray[np.uint16]] | None:
        """Turn each mark's channel names into bundle channel indices.

        A mark naming no channels keeps an empty array, which is how the format
        says a mark belongs to the recording rather than to named electrodes.
        """
        if not any("channels" in mark for mark in marks):
            return None
        if index_by_name is None:
            raise AnnotationDocumentError(
                "this document names channels, so it needs the bundle's "
                "channel names to resolve them against"
            )
        resolved: list[npt.NDArray[np.uint16]] = []
        for mark in marks:
            names = mark.get("channels", [])
            unknown = [name for name in names if name not in index_by_name]
            if unknown:
                raise AnnotationDocumentError(
                    f"channels {unknown} are not in this bundle"
                )
            resolved.append(
                np.array(
                    [index_by_name[name] for name in names], dtype=np.uint16
                )
            )
        return resolved

    @property
    def id(self) -> str:
        """The document's channel id."""
        return self._id

    @property
    def name(self) -> str:
        """The channel's display label."""
        return self._name

    @property
    def unit(self) -> str:
        """The unit of the per-mark value, when the document declares one."""
        return self._unit

    def start_us(self) -> int:
        """Return the wall-clock microseconds of the first mark."""
        return int(self._events[0]) if self._events.size else 0

    def num_events(self) -> int:
        """Return the number of marks."""
        return int(self._events.shape[0])

    def num_labels(self) -> int:
        """Return the size of the label space, 0 when nothing is labelled."""
        return len(self._label_names) if self._label_names else 0

    def label_names(self) -> list[str] | None:
        """Return the label names, index-aligned with the labels column."""
        return self._label_names

    def max_duration_us(self) -> int | None:
        """Return a bound on the durations, or None when the marks are points.

        The writer recomputes this from what it writes; this only declares that
        the column is there.
        """
        if self._durations is None:
            return None
        return int(self._durations.max()) if self._durations.size else 0

    def body_media_type(self) -> str | None:
        """Return the payload media type, or None when there are no payloads."""
        return self._media_type

    def has_values(self) -> bool:
        """Whether any mark carries a measurement."""
        return self._values is not None

    def has_channel_refs(self) -> bool:
        """Whether any mark names the channels it belongs to."""
        return self._refs is not None

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of absolute-microsecond timestamps."""
        return self._events[start:stop]

    def read_durations(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the [start, stop) window of durations in microseconds."""
        assert self._durations is not None
        return self._durations[start:stop]

    def read_labels(self, start: int, stop: int) -> npt.NDArray[np.uint16]:
        """Return the [start, stop) window of label indices."""
        assert self._labels is not None
        return self._labels[start:stop]

    def read_values(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the [start, stop) window of per-mark measurements."""
        assert self._values is not None
        return self._values[start:stop]

    def read_bodies(self, start: int, stop: int) -> list[bytes]:
        """Return one encoded payload per mark in [start, stop)."""
        assert self._bodies is not None
        return self._bodies[start:stop]

    def read_channel_refs(
        self, start: int, stop: int
    ) -> list[npt.NDArray[np.uint16]]:
        """Return the channel indices each mark in [start, stop) applies to."""
        assert self._refs is not None
        return self._refs[start:stop]


def load_annotation_document(path: Path) -> dict[str, Any]:
    """Read one annotation document from disk.

    Raises AnnotationDocumentError for anything that is not a JSON object, so a
    sketch with comments in it fails here rather than halfway through a channel.
    """
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise AnnotationDocumentError(
            f"{path.name} is not valid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise AnnotationDocumentError(
            f"{path.name} holds a {type(document).__name__}, not an object"
        )
    return document


def build_annotation_sources(
    paths: list[Path],
    recording_onset_us: int,
    *,
    channel_index_by_name: dict[str, int] | None = None,
) -> list[JsonAnnotationSource]:
    """Return one source per document, in the order the paths were given.

    One document is one channel, so the order here is the order the annotation
    channels are indexed in.
    """
    return [
        JsonAnnotationSource(
            load_annotation_document(path),
            recording_onset_us,
            channel_index_by_name=channel_index_by_name,
        )
        for path in paths
    ]
