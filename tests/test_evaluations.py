"""Ground-truth listening notes: issues.yaml integrity and the notes -> YAML draft parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "docs" / "evaluations"
CATEGORIES = {
    "sibilance", "consonant_attack", "step_change", "buried", "sits_on_top", "thin_light", "muddy",
    "airy_fizzy", "space_excess", "delay_offbeat", "fx_dropout", "harmony_prominent", "distance",
    "harmony_missing",
}


def _load_parser():
    spec = importlib.util.spec_from_file_location("parse_notes", ROOT / "scripts" / "evaluations" / "parse_notes.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_issues_yaml_is_consistent():
    doc = yaml.safe_load((EVAL_DIR / "issues.yaml").read_text(encoding="utf-8"))
    sources: dict[str, str] = {}

    def source_of(it: dict) -> str:
        rel = it.get("source", doc["source"])  # an item may come from another verbatim note file
        if rel not in sources:
            sources[rel] = (ROOT / rel).read_text(encoding="utf-8")
        return sources[rel]

    ids = [it["id"] for it in doc["issues"]]
    assert len(ids) == len(set(ids))
    for it in doc["issues"]:
        assert it["category"] in CATEGORIES, it["id"]
        assert ("time_sec" in it) != (it.get("scope") == "global"), it["id"]
        for part in it["quote"].split(" / "):
            assert part in source_of(it), it["id"]  # quotes are the user's words verbatim


def test_parse_notes_basic():
    pn = _load_parser()
    notes = "・エバ\n全体的に\n声がオケに負けている\n\n0:20～「真の髄から」の「し」の歯擦音がつぶれてる\n2:24、2:26のディレイが拍に合ってない\n"
    issues = pn.parse_notes(notes, known={})
    assert [i.get("time_sec", "g") for i in issues] == ["g", 20, 144, 146]
    assert issues[0]["scope"] == "global" and issues[0]["category"] == "buried"
    assert issues[1]["lyric"] == "真の髄から"
    assert (issues[1]["category"], issues[1]["direction"]) == ("sibilance", "crushed")
    assert issues[2]["category"] == "delay_offbeat"
    assert all(i["draft"] for i in issues)


def test_parse_notes_maps_song_alias():
    pn = _load_parser()
    known = {pn._norm("ラブ&デストロイ"): "ラブ&デストロイ"}
    issues = pn.parse_notes("・ラブアンドデストロイ\n0:46の「悪さを手に入れたい」の「さ」がつぶれてる\n", known=known)
    assert issues[0]["song"] == "ラブ&デストロイ"
