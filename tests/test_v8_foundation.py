"""Phase 1 (v8 foundation): engine switch and render provenance."""

from __future__ import annotations

import hashlib

import pytest
from typer.testing import CliRunner

from mix_automation import provenance
from mix_automation.cli import app
from mix_automation.v8 import V8NotImplementedError, run_v8_automix_for_song

runner = CliRunner()


def test_engine_v8_stops_with_not_implemented(tmp_path):
    result = runner.invoke(app, ["mix", str(tmp_path), "--engine", "v8", "-o", str(tmp_path / "out")])
    assert result.exit_code == 3
    assert "未実装" in result.output
    assert not (tmp_path / "out").exists()


def test_unknown_engine_is_rejected(tmp_path):
    result = runner.invoke(app, ["mix", str(tmp_path), "--engine", "v9"])
    assert result.exit_code == 2
    assert "Unknown --engine" in result.output


def test_v8_entry_point_raises(tmp_path):
    with pytest.raises(V8NotImplementedError):
        run_v8_automix_for_song(tmp_path, tmp_path / "t.song", tmp_path / "p.json", tmp_path)


def test_render_provenance_fields(tmp_path):
    prof = tmp_path / "profiles.json"
    prof.write_bytes(b'{"a": 1}')
    info = provenance.render_provenance("v7", prof)
    assert info["engine"] == "v7"
    assert info["profiles_sha256"] == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert info["profiles_json"] == str(prof)
    commit = info["git_commit"]
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)
    assert info["git_dirty"] in (True, False, None)


def test_file_sha256_missing_file(tmp_path):
    assert provenance.file_sha256(tmp_path / "missing.json") is None
    assert provenance.file_sha256(None) is None
