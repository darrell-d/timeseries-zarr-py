import json

import numpy as np
from zarr import Group
from zarr.codecs import ShardingCodec, ZstdCodec

from timeseries_zarr.zarr_io import (
    consolidate,
    create_array,
    create_group_with_attrs,
    open_group,
    write_region,
)


def test_open_group_creates_when_absent(tmp_path):
    path = tmp_path / "bundle"
    assert not path.exists()
    group = open_group(path)
    assert isinstance(group, Group)
    assert path.exists()


def test_open_group_is_v3(tmp_path):
    group = open_group(tmp_path / "bundle")
    assert group.metadata.zarr_format == 3


def test_open_group_opens_existing_in_place(tmp_path):
    path = tmp_path / "bundle"
    open_group(path).attrs["marker"] = "kept"
    reopened = open_group(path)
    assert reopened.attrs["marker"] == "kept"


def test_create_group_with_attrs_creates_child(tmp_path):
    path = tmp_path / "bundle"
    root = open_group(path)
    child = create_group_with_attrs(root, "0", {})
    assert isinstance(child, Group)
    assert "0" in open_group(path)


def test_create_group_with_attrs_sets_attrs_verbatim(tmp_path):
    path = tmp_path / "bundle"
    attrs = {
        "id": "N:ch:abc",
        "rate_hz": 32000.0,
        "start_us": 1000,
        "kind": "continuous",
    }
    create_group_with_attrs(open_group(path), "0", attrs)
    assert dict(open_group(path)["0"].attrs) == attrs


def test_create_group_with_attrs_empty_attrs(tmp_path):
    path = tmp_path / "bundle"
    create_group_with_attrs(open_group(path), "0", {})
    assert dict(open_group(path)["0"].attrs) == {}


def test_create_array_shape_and_dtype(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), {}, 5
    )
    arr = open_group(path)["lvl"]
    assert arr.shape == (1000,)
    assert arr.dtype == np.float32


def test_create_array_chunk_and_shard_grid(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), {}, 5
    )
    arr = open_group(path)["lvl"]
    assert arr.chunks == (256,)
    assert arr.shards == (512,)


def test_create_array_rank2_keeps_envelope_axis(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path),
        "lvl",
        (500, 2),
        np.float32,
        (256, 2),
        (512, 2),
        {},
        5,
    )
    arr = open_group(path)["lvl"]
    assert arr.shape == (500, 2)
    assert arr.chunks == (256, 2)
    assert arr.shards == (512, 2)


def test_create_array_uses_zstd_at_level(tmp_path):
    path = tmp_path / "bundle"
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), {}, 7
    )
    arr = open_group(path)["lvl"]
    sharding = arr.metadata.codecs[0]
    assert isinstance(sharding, ShardingCodec)
    assert any(
        isinstance(c, ZstdCodec) and c.level == 7 for c in sharding.codecs
    )


def test_create_array_sets_attrs_verbatim(tmp_path):
    path = tmp_path / "bundle"
    attrs = {"period_us": 31.25}
    create_array(
        open_group(path), "lvl", (1000,), np.float32, (256,), (512,), attrs, 5
    )
    assert dict(open_group(path)["lvl"].attrs) == attrs


def _make_array(path, shape):
    chunk = (4, *shape[1:])
    shard = (8, *shape[1:])
    return create_array(
        open_group(path), "lvl", shape, np.float32, chunk, shard, {}, 5
    )


def test_write_region_rank1_at_offset(tmp_path):
    path = tmp_path / "bundle"
    arr = _make_array(path, (10,))
    block = np.array([1, 2, 3, 4], dtype=np.float32)
    write_region(arr, 2, block)
    assert np.array_equal(open_group(path)["lvl"][2:6], block)


def test_write_region_rank2_envelope(tmp_path):
    path = tmp_path / "bundle"
    arr = _make_array(path, (10, 2))
    block = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
    write_region(arr, 1, block)
    assert np.array_equal(open_group(path)["lvl"][1:4], block)


def test_write_region_multiple_regions_tile_full_array(tmp_path):
    path = tmp_path / "bundle"
    arr = _make_array(path, (10,))
    full = np.arange(10, dtype=np.float32)
    write_region(arr, 0, full[:6])
    write_region(arr, 6, full[6:])
    assert np.array_equal(open_group(path)["lvl"][:], full)


def _build_tree(path):
    root = open_group(path)
    child = create_group_with_attrs(root, "0", {"kind": "continuous"})
    create_array(child, "lvl", (8,), np.float32, (4,), (8,), {}, 5)
    return root


def test_consolidate_writes_inline_block(tmp_path):
    path = tmp_path / "bundle"
    consolidate(_build_tree(path))
    root_json = json.loads((path / "zarr.json").read_text())
    members = root_json["consolidated_metadata"]["metadata"]
    assert set(members) == {"0", "0/lvl"}


def test_consolidate_no_v2_sidecar(tmp_path):
    path = tmp_path / "bundle"
    consolidate(_build_tree(path))
    assert not (path / ".zmetadata").exists()


def test_consolidate_returns_none(tmp_path):
    path = tmp_path / "bundle"
    assert consolidate(_build_tree(path)) is None
