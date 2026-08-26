"""Output properties written beside the published bundle."""

import json
from pathlib import Path
from typing import Final

DEFAULT_PROPERTIES_FILE: Final = "asset-properties.json"
"""Properties file name used when ASSET_PROPERTIES_FILE is unset."""

ROOT_PATH_KEY: Final = "root_path"
"""Properties key holding the published bundle's directory name."""


def write_properties(properties_path: Path, bundle_dir: Path) -> None:
    """Write the output properties naming the published bundle.

    root_path holds the bundle directory's name, resolved against the
    directory holding the properties file, so the two paths share a parent.
    An existing file is overwritten.
    """
    properties_path.write_text(
        json.dumps({ROOT_PATH_KEY: bundle_dir.name}) + "\n", encoding="utf-8"
    )
