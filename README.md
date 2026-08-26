# timeseries-zarr-py

Converts a neurophysiology recording in NWB into a pyramid Zarr v3 viewer bundle: a
static directory that a browser reads over HTTP range requests to render any time window
interactively.

A raw recording runs to hundreds of gigabytes, far more than a browser can fetch to draw
an overview 2000 pixels wide. This writer precomputes multi-resolution min/max envelopes
at ingest so a reader fetches only the bins one viewport needs. It writes the format and
nothing else: no rendering, no reader.

The format is specified in [docs/bundle-format.md](./docs/bundle-format.md). The reference
reader is
[`@pennsieve/timeseries-zarr-reader`](https://github.com/Pennsieve/timeseries-zarr-reader).

## Usage

Read one NWB file and write one bundle:

```bash
python -m timeseries_zarr.main recording.nwb /data/recording.zarr
```

With no arguments the writer takes the directory convention instead: it reads the single
`.nwb` file in `INPUT_DIR` and publishes to `OUTPUT_DIR/<input-stem>.zarr`. This is how
the container runs.

Every run also writes a sibling properties file next to the bundle,
`asset-properties.json` by default (`ASSET_PROPERTIES_FILE` overrides the name),
recording the bundle's directory name under the key `root_path`.

```bash
make run        # docker-compose build + up, against data/input and data/output
```

Writer settings come from the environment under the `ZARR_WRITER_` prefix. All are
optional and fall back to the format defaults: `ZARR_WRITER_STAGING_DIR` (scratch path for
the atomic publish, by default alongside the output), `ZARR_WRITER_ZSTD_LEVEL`,
`ZARR_WRITER_MAX_LEVELS`, `ZARR_WRITER_MIN_BINS`, `ZARR_WRITER_INNER_LEN`, and
`ZARR_WRITER_TARGET_SHARD_BYTES`.

## Development

Python 3.12, fully typed under `mypy --strict`, with a strict `ruff` ruleset. Tests in
`tests/` mirror `timeseries_zarr/` one to one.

```bash
make venv        # create the virtualenv and install deps
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (rewrites files)
make check       # the gate: ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

`make check` must stay green.

Two conventions the linters do not catch. Imports are absolute only, which ruff does
enforce. `zarr` is imported in `zarr_io.py` and nowhere else, which it does not.

The module layout and data flow are in [docs/architecture.md](./docs/architecture.md).

## Dependencies

Runtime: `zarr>=3`, `numcodecs`, `numpy`, `pynwb`, `h5py`. Development: `pytest`,
`pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `pre-commit`.

## License

Apache-2.0. See [LICENSE](LICENSE).
