"""Real-photo evaluation set: Wikimedia Commons photos of the 62 trained people.

The images are not stored in git. data/real_photos/manifest.json lists each
photo's URL, SHA-256, license, and author, and download_images() fetches the
files into data/real_photos/images/. scripts/collect_real_photos.py builds the
manifest; docs/REAL_PHOTOS.md describes how the photos were chosen.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

REAL_PHOTOS_DIR = Path(__file__).resolve().parent.parent / "data" / "real_photos"
PEOPLE_FILE = REAL_PHOTOS_DIR / "people.json"
MANIFEST_FILE = REAL_PHOTOS_DIR / "manifest.json"
EXCLUDED_FILE = REAL_PHOTOS_DIR / "excluded.json"
IMAGES_DIR = REAL_PHOTOS_DIR / "images"

# Wikimedia asks automated clients for a descriptive User-Agent.
USER_AGENT = "real-photo-eval/0.1 (https://github.com/lmmh202/face-recognition-serving)"
REQUEST_INTERVAL_SECONDS = 1.0
MAX_ATTEMPTS = 10


def http_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)


def polite_get(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    """GET with a pause after each request, backing off on 429, 5xx, and dropped connections."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError as error:
            print(f"{error!r}, retrying in {5 * (attempt + 1)}s: {url}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            wait = float(response.headers.get("retry-after", 5 * (attempt + 1))) + 1
            print(f"HTTP {response.status_code}, retrying in {wait:.0f}s: {url}", file=sys.stderr)
            time.sleep(wait)
            continue
        response.raise_for_status()
        time.sleep(REQUEST_INTERVAL_SECONDS)
        return response
    raise RuntimeError(f"giving up on {url} after {MAX_ATTEMPTS} attempts")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))["photos"]


def image_path(photo: dict) -> Path:
    return IMAGES_DIR / photo["file"]


def download_images(photos: list[dict]) -> None:
    """Fetch missing images and check every file against its recorded SHA-256."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with http_client() as client:
        for photo in photos:
            path = image_path(photo)
            if not path.exists():
                path.write_bytes(polite_get(client, photo["image_url"]).content)
            if sha256(path.read_bytes()) != photo["sha256"]:
                raise ValueError(f"{path.name}: SHA-256 differs from the manifest")
