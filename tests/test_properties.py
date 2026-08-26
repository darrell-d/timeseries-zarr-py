import json

from timeseries_zarr.properties import ROOT_PATH_KEY, write_properties


def test_declares_the_bundle_directory_name(tmp_path):
    properties_path = tmp_path / "asset-properties.json"

    write_properties(properties_path, tmp_path / "session.zarr")

    assert json.loads(properties_path.read_text()) == {
        ROOT_PATH_KEY: "session.zarr"
    }


def test_declares_a_name_not_a_path(tmp_path):
    properties_path = tmp_path / "asset-properties.json"

    write_properties(properties_path, tmp_path / "nested" / "session.zarr")

    assert (
        json.loads(properties_path.read_text())[ROOT_PATH_KEY] == "session.zarr"
    )


def test_overwrites_an_existing_file(tmp_path):
    properties_path = tmp_path / "asset-properties.json"
    properties_path.write_text('{"root_path": "stale.zarr"}')

    write_properties(properties_path, tmp_path / "fresh.zarr")

    assert (
        json.loads(properties_path.read_text())[ROOT_PATH_KEY] == "fresh.zarr"
    )


def test_writes_one_trailing_newline(tmp_path):
    properties_path = tmp_path / "asset-properties.json"

    write_properties(properties_path, tmp_path / "session.zarr")

    text = properties_path.read_text()
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
