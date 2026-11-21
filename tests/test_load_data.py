import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_data


def test_load_properties_uses_default_path(monkeypatch):
    monkeypatch.delenv("CURATED_DATA_PATH", raising=False)
    expected = pd.DataFrame({"preco": [100]})
    paths = []
    monkeypatch.setattr(
        load_data.pd, "read_parquet", lambda path: paths.append(path) or expected
    )

    assert load_data.load_properties() is expected
    assert paths == [
        Path(load_data.__file__).resolve().parent.parent
        / "data"
        / "curated"
        / "properties.parquet"
    ]


def test_load_properties_uses_configured_path(tmp_path, monkeypatch):
    path = tmp_path / "curated.parquet"
    expected = pd.DataFrame({"preco": [123, 456]})
    expected.to_parquet(path, index=False)
    monkeypatch.setenv("CURATED_DATA_PATH", str(path))

    pd.testing.assert_frame_equal(load_data.load_properties(), expected)


def test_load_properties_rejects_empty_configured_path(monkeypatch):
    monkeypatch.setenv("CURATED_DATA_PATH", "")

    with pytest.raises(ValueError, match="CURATED_DATA_PATH"):
        load_data.load_properties()
