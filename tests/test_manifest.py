"""Tests for manifest reading -- no S3 or torch required."""

from __future__ import annotations

import json

import pytest

from gpulab.data.dataset import read_manifest


def test_json_manifest_with_id_key(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps(
            [
                {"id": "260128_D06...C1A_...-Phusion", "species": "K. pneumoniae"},
                {"id": "260128_D06...C4A_...-Phusion", "species": "E. coli"},
            ]
        ),
        encoding="utf-8",
    )
    df = read_manifest(p)
    assert list(df.columns) == ["chip_id", "species"]
    assert df.shape == (2, 2)
    assert df.loc[0, "chip_id"] == "260128_D06...C1A_...-Phusion"
    assert set(df["species"]) == {"K. pneumoniae", "E. coli"}


def test_json_manifest_accepts_chip_id_key(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps([{"chip_id": "abc", "species": "E. coli"}]), encoding="utf-8")
    assert read_manifest(p).loc[0, "chip_id"] == "abc"


def test_json_skips_incomplete_rows(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps([{"id": "ok", "species": "E. coli"}, {"id": "no_species"}]),
        encoding="utf-8",
    )
    assert read_manifest(p).shape == (1, 2)


def test_csv_manifest_chip_id_column(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text("chip_id,species\nabc,E. coli\ndef,S. aureus\n", encoding="utf-8")
    df = read_manifest(p)
    assert df.shape == (2, 2)
    assert list(df["chip_id"]) == ["abc", "def"]


def test_csv_manifest_id_column(tmp_path):
    p = tmp_path / "manifest.csv"
    p.write_text("id,species\nabc,E. coli\n", encoding="utf-8")
    assert read_manifest(p).loc[0, "chip_id"] == "abc"


def test_empty_manifest_raises(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(p)


def test_example_json_is_valid(tmp_path):
    # The committed example must stay parseable in the documented format.
    df = read_manifest("configs/manifest.example.json")
    assert list(df.columns) == ["chip_id", "species"]
    assert len(df) >= 1
