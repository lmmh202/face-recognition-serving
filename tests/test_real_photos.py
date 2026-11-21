"""Consistency of the committed real-photo manifest; no network access."""

import json
import re

from ml.real_photos import MANIFEST_FILE, PEOPLE_FILE

THUMBNAIL = re.compile(r"^https://upload\.wikimedia\.org/wikipedia/commons/thumb/.+/(250|330|500|960|1280)px-[^/]+$")


def manifest() -> dict:
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def test_people_map_to_distinct_wikidata_items():
    people = json.loads(PEOPLE_FILE.read_text(encoding="utf-8"))
    assert len(people) == 62
    assert all(re.fullmatch(r"Q\d+", qid) for qid in people.values())
    assert len(set(people.values())) == 62


def test_photos_are_labelled_unique_and_standard_thumbnails():
    people = json.loads(PEOPLE_FILE.read_text(encoding="utf-8"))
    photos = manifest()["photos"]
    assert photos
    assert {p["label"] for p in photos} <= set(people)
    assert len({p["file"] for p in photos}) == len(photos)
    assert len({p["sha256"] for p in photos}) == len(photos)
    for photo in photos:
        assert THUMBNAIL.match(photo["image_url"]), photo["image_url"]
        assert re.fullmatch(r"[0-9a-f]{64}", photo["sha256"])
        assert photo["label_source"] in {"depicts", "category"}
        assert photo["license"] and photo["page_url"].startswith("https://commons.wikimedia.org/")


def test_per_person_counts_match_photos():
    data = manifest()
    per_person = data["selection"]["per_person"]
    for name, stats in data["people"].items():
        chosen = [p for p in data["photos"] if p["label"] == name]
        assert len(chosen) == stats["chosen"] <= per_person
