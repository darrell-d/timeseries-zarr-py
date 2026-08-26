from timeseries_zarr.constants import UNIT_TO_UV


def test_unit_to_uv_factors():
    assert UNIT_TO_UV["volts"] == 1e6
    assert UNIT_TO_UV["v"] == 1e6
    assert UNIT_TO_UV["millivolts"] == 1e3
    assert UNIT_TO_UV["mv"] == 1e3
    assert UNIT_TO_UV["microvolts"] == 1.0
    assert UNIT_TO_UV["uv"] == 1.0


def test_unit_to_uv_aliases_agree():
    assert UNIT_TO_UV["v"] == UNIT_TO_UV["volts"]
    assert UNIT_TO_UV["mv"] == UNIT_TO_UV["millivolts"]
    assert UNIT_TO_UV["uv"] == UNIT_TO_UV["microvolts"]


def test_unit_to_uv_keys_lowercase():
    assert all(k == k.lower() for k in UNIT_TO_UV)
