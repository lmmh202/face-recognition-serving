"""Choose Wikimedia Commons photos of the 62 trained people and write the manifest.

Run with: uv run python scripts/collect_real_photos.py [--per-person 5]

For each person, Commons files whose structured data says they depict that
person (P180) are shuffled with a fixed seed and checked one by one until
--per-person photos pass. If that leaves the person short, files in their
Commons category (Wikidata P373) are checked the same way; those photos are
recorded with label_source "category" instead of "depicts". A photo is kept
only if:

- P180 lists that person and no one else (for category files, P180 may also
  be empty),
- it is a JPEG or PNG and nothing in its title or categories marks it as a
  painting, statue, wax figure, or other non-photograph,
- its title does not describe several people ("X and Y", "X meets Y"),
  unless it is a "(cropped)" derivative, where the crop isolates one person,
- YuNet finds one clear face: no second face at least half the size of the
  largest, and the largest is at least MIN_FACE_SIDE pixels,
- its detected face crop is not a near duplicate of any LFW image of that
  person or of a photo already chosen,
- no photo of that person taken on the same day was chosen already, so one
  event does not supply several near-identical shots,
- it fits the service's upload limits (MAX_BODY_BYTES, MAX_PIXELS), since
  /predict would reject it with 413,
- it is not in data/real_photos/excluded.json, which lists photos that passed
  these rules but turned out, on a look at the images, not to be a usable
  photograph of one person (a painting, a playing card, a group shot).

The manifest is rewritten after each person, and a rerun skips the people it
already lists, so an interrupted run resumes where it stopped. Each person's
candidates are shuffled with their own RandomState(SEED), so a resumed run
picks what a single run would. Search results change over time, though, so a
fresh run on another day may pick a different set; the committed manifest is
the evaluation set. Pass --restart to discard the manifest and start over.
"""

import argparse
import datetime
import io
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.main import MAX_BODY_BYTES, MAX_PIXELS
from ml.face_crop import DEFAULT_FACE_CROP, YUNET_MODEL, FaceCropper
from ml.preprocess import preprocess
from ml.real_photos import (
    EXCLUDED_FILE,
    IMAGES_DIR,
    REAL_PHOTOS_DIR,
    MANIFEST_FILE,
    PEOPLE_FILE,
    http_client,
    polite_get,
    sha256,
)
from ml.train import lfw_person_dirs

CACHE_DIR = REAL_PHOTOS_DIR / "cache"  # every downloaded candidate, so reruns skip the download
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SEED = 0
SEARCH_LIMIT = 500  # candidates per person, in Commons relevance order
BATCH = 50  # Commons API limit for ids per request
# upload.wikimedia.org rejects automated requests for originals and for
# non-standard thumbnail widths (https://www.mediawiki.org/wiki/Common_thumbnail_sizes).
THUMB_WIDTHS = (250, 330, 500, 960, 1280)
MIME_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}

COUNT_SCORE = 0.5  # lower than serving (0.6) so faint second faces still count
SECOND_FACE_RATIO = 0.5
MIN_FACE_SIDE = 64
DUPLICATE_CORRELATION = 0.95

NON_PHOTO = re.compile(
    r"wax|statue|sculpt|bust of|painting|drawing|caricatur|cartoon|mural|graffiti|"
    r"stamp|coin|banknote|poster|mask|puppet|figurine|illustration|sketch|"
    r"lookalike|look-alike|impersonat|cosplay|screenshot|signature|autograph",
    re.IGNORECASE,
)
# Titles like "Bush and Cheney in the Oval Office" or "Aznar impone al diputado
# Múgica ...": the one large face found may belong to the other person.
SEVERAL_PEOPLE = re.compile(
    r"\b(and|with|y|e|et|und|con|mit|avec|meets?|meeting|met|greets?|receives?|"
    r"visits?|toured|impone|signing|attended)\b",
    re.IGNORECASE,
)
CROPPED = re.compile(r"\(cropped\)", re.IGNORECASE)
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
TAG = re.compile(r"<[^>]+>")
TIME_OF_DAY = re.compile(r",?\s*\d{1,2}:\d{2}.*$")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def clean(html: str) -> str:
    return " ".join(TAG.sub("", html).split())


def search_depicting(client, qid: str) -> list[int]:
    data = polite_get(client, COMMONS_API, {
        "action": "query", "list": "search", "srsearch": f"haswbstatement:P180={qid}",
        "srnamespace": 6, "srlimit": SEARCH_LIMIT, "format": "json",
    }).json()
    return sorted(result["pageid"] for result in data["query"]["search"])


def commons_category(client, qid: str) -> str | None:
    claims = polite_get(client, WIKIDATA_API, {
        "action": "wbgetclaims", "entity": qid, "property": "P373", "format": "json",
    }).json().get("claims", {}).get("P373", [])
    return claims[0]["mainsnak"]["datavalue"]["value"] if claims else None


def category_files(client, category: str) -> list[int]:
    data = polite_get(client, COMMONS_API, {
        "action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
        "cmtype": "file", "cmlimit": SEARCH_LIMIT, "format": "json",
    }).json()
    return sorted(member["pageid"] for member in data["query"]["categorymembers"])


def depicts(client, pageids: list[int]) -> dict[int, list[str]]:
    data = polite_get(client, COMMONS_API, {
        "action": "wbgetentities", "ids": "|".join(f"M{p}" for p in pageids), "format": "json",
    }).json()
    result = {}
    for entity_id, entity in data.get("entities", {}).items():
        statements = entity.get("statements") or {}
        result[int(entity_id[1:])] = [
            s["mainsnak"]["datavalue"]["value"]["id"]
            for s in statements.get("P180", [])
            if s["mainsnak"].get("datavalue")
        ]
    return result


def image_info(client, pageids: list[int]) -> dict[int, dict]:
    data = polite_get(client, COMMONS_API, {
        "action": "query", "pageids": "|".join(map(str, pageids)), "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|Artist|DateTimeOriginal|Categories",
        "format": "json",
    }).json()
    return {
        int(pageid): {"title": page["title"], **page["imageinfo"][0]}
        for pageid, page in data["query"]["pages"].items()
        if page.get("imageinfo")
    }


def thumbnail_url(original_url: str, original_width: int) -> str | None:
    """The largest standard thumbnail narrower than the original, or None if it is too small.

    /wikipedia/commons/a/ab/Name.jpg -> /wikipedia/commons/thumb/a/ab/Name.jpg/960px-Name.jpg
    """
    widths = [w for w in THUMB_WIDTHS if w < original_width]
    if not widths:
        return None
    parts = urlsplit(original_url)
    prefix, rest = parts.path.split("/commons/", 1)
    filename = rest.rsplit("/", 1)[1]
    path = f"{prefix}/commons/thumb/{rest}/{widths[-1]}px-{filename}"
    return urlunsplit(parts._replace(path=path, query=""))


class FaceCounter:
    def __init__(self):
        self._detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), COUNT_SCORE)

    def sides(self, image: Image.Image) -> list[float]:
        """Longer side of each detected face, in original pixels, largest first."""
        scale = min(1.0, 640 / max(image.size))
        small = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        self._detector.setInputSize(small.size)
        _, faces = self._detector.detect(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        boxes = faces[:, 2:4]
        boxes = boxes[np.isfinite(boxes).all(axis=1) & (boxes > 0).all(axis=1)]
        return sorted((float(box.max()) / scale for box in boxes), reverse=True)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def lfw_faces(name: str, cropper: FaceCropper) -> list[np.ndarray]:
    directory = next(d for d in lfw_person_dirs() if d.name == name.replace(" ", "_"))
    faces = []
    for path in sorted(directory.iterdir()):
        with Image.open(path) as image:
            cropped = cropper(image)
        if cropped is not None:
            faces.append(preprocess(cropped))
    return faces


def check(image: Image.Image, counter: FaceCounter, cropper: FaceCropper,
          references: list[np.ndarray], chosen: list[np.ndarray]) -> tuple[str, np.ndarray | None]:
    sides = counter.sides(image)
    if not sides:
        return "no_face", None
    if len(sides) > 1 and sides[1] >= SECOND_FACE_RATIO * sides[0]:
        return "several_faces", None
    if sides[0] < MIN_FACE_SIDE:
        return "small_face", None
    cropped = cropper(image)
    if cropped is None:
        return "no_face", None
    face = preprocess(cropped)
    if any(correlation(face, ref) > DUPLICATE_CORRELATION for ref in references):
        return "lfw_duplicate", None
    if any(correlation(face, other) > DUPLICATE_CORRELATION for other in chosen):
        return "duplicate", None
    return "ok", face


class Person:
    """Selection state for one person across both candidate sources."""

    def __init__(self, name: str, qid: str, per_person: int, cropper: FaceCropper):
        self.name, self.qid, self.per_person = name, qid, per_person
        self.reasons = Counter()
        self.references = lfw_faces(name, cropper)
        self.photos, self.faces, self.days, self.seen = [], [], set(), set()

    @property
    def full(self) -> bool:
        return len(self.photos) >= self.per_person


def consider(client, person: Person, candidates: list[int], source: str, counter, cropper) -> None:
    candidates = [p for p in candidates if p not in person.seen]
    person.seen.update(candidates)
    np.random.RandomState(SEED).shuffle(candidates)
    allowed = ([person.qid],) if source == "depicts" else ([person.qid], [])
    for start in range(0, len(candidates), BATCH):
        if person.full:
            return
        batch = candidates[start:start + BATCH]
        subjects = depicts(client, batch)
        sole = [p for p in batch if subjects.get(p, []) in allowed]
        person.reasons["other_people_depicted"] += len(batch) - len(sole)
        infos = image_info(client, sole) if sole else {}
        for pageid in sole:
            if person.full:
                return
            reason = check_candidate(client, person, pageid, infos.get(pageid), source, counter, cropper)
            person.reasons[reason] += 1
            if reason not in {"other_people_depicted", "not_jpeg_or_png", "not_a_photograph",
                              "title_names_others", "same_day", "too_small"}:
                print(f"  {person.name} {pageid} ({source}): {reason}", flush=True)


def excluded() -> set[int]:
    data = json.loads(EXCLUDED_FILE.read_text(encoding="utf-8"))
    return {int(pageid) for pageid in data if not pageid.startswith("_")}


EXCLUDED = excluded()


def check_candidate(client, person: Person, pageid: int, info: dict | None, source: str,
                    counter, cropper) -> str:
    if pageid in EXCLUDED:
        return "excluded_after_review"
    if info is None or info["mime"] not in MIME_TYPES:
        return "not_jpeg_or_png"
    meta = info.get("extmetadata", {})
    title = info["title"]
    if NON_PHOTO.search(f"{title} {meta.get('Categories', {}).get('value', '')}"):
        return "not_a_photograph"
    if SEVERAL_PEOPLE.search(title.replace("_", " ")) and not CROPPED.search(title):
        return "title_names_others"
    taken = clean(meta.get("DateTimeOriginal", {}).get("value", ""))
    day = TIME_OF_DAY.sub("", taken).strip()
    if day and day in person.days:
        return "same_day"
    url = thumbnail_url(info["url"], info["width"])
    if url is None:
        return "too_small"
    cached = CACHE_DIR / f"{pageid}{MIME_TYPES[info['mime']]}"
    if cached.exists():
        data = cached.read_bytes()
    else:
        data = polite_get(client, url).content
        cached.write_bytes(data)
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if len(data) > MAX_BODY_BYTES or image.width * image.height > MAX_PIXELS:
        return "over_upload_limit"
    reason, face = check(image, counter, cropper, person.references, person.faces)
    if reason != "ok":
        return reason
    file = f"{slug(person.name)}_{pageid}{MIME_TYPES[info['mime']]}"
    (IMAGES_DIR / file).write_bytes(data)
    year = YEAR.search(taken)
    person.photos.append({
        "label": person.name,
        "label_source": source,
        "file": file,
        "image_url": url,
        "sha256": sha256(data),
        "page_url": info["descriptionurl"],
        "license": clean(meta.get("LicenseShortName", {}).get("value", "")),
        "artist": clean(meta.get("Artist", {}).get("value", "")),
        "year": int(year.group()) if year else None,
    })
    person.faces.append(face)
    if day:
        person.days.add(day)
    return "ok"


def collect_person(client, name: str, qid: str, per_person: int, counter, cropper):
    person = Person(name, qid, per_person, cropper)
    from_depicts = search_depicting(client, qid)
    consider(client, person, from_depicts, "depicts", counter, cropper)
    category, from_category = None, []
    if not person.full:
        category = commons_category(client, qid)
        if category:
            from_category = category_files(client, category)
            consider(client, person, from_category, "category", counter, cropper)
    sources = {"depicts": len(from_depicts), "category": category, "category_files": len(from_category)}
    return person.photos, person.reasons, sources


def selection(per_person: int) -> dict:
    return {
        "seed": SEED,
        "per_person": per_person,
        "search_limit": SEARCH_LIMIT,
        "thumb_widths": list(THUMB_WIDTHS),
        "count_score": COUNT_SCORE,
        "second_face_ratio": SECOND_FACE_RATIO,
        "min_face_side": MIN_FACE_SIDE,
        "duplicate_correlation": DUPLICATE_CORRELATION,
        "several_people_title": SEVERAL_PEOPLE.pattern,
        "category_fallback": True,
        "excluded": sorted(EXCLUDED),
        "upload_limits": [MAX_BODY_BYTES, MAX_PIXELS],
    }


def write_manifest(settings: dict, stats: dict, photos: list[dict], complete: bool) -> None:
    MANIFEST_FILE.write_text(json.dumps({
        "created": datetime.date.today().isoformat(),
        "complete": complete,
        "source": "Wikimedia Commons, files whose structured data depicts (P180) the person",
        "selection": settings,
        "people": stats,
        "photos": photos,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-person", type=int, default=5)
    parser.add_argument("--restart", action="store_true", help="discard the existing manifest")
    args = parser.parse_args()

    people = json.loads(PEOPLE_FILE.read_text(encoding="utf-8"))
    settings = selection(args.per_person)
    photos, stats = [], {}
    if MANIFEST_FILE.exists() and not args.restart:
        previous = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if previous["selection"] != settings:
            raise SystemExit("manifest was made with other settings; pass --restart")
        photos, stats = previous["photos"], previous["people"]
        print(f"resuming after {len(stats)} people", flush=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    counter = FaceCounter()
    cropper = FaceCropper(DEFAULT_FACE_CROP)
    with http_client() as client:
        for name, qid in people.items():
            if name in stats:
                continue
            chosen, reasons, sources = collect_person(
                client, name, qid, args.per_person, counter, cropper
            )
            photos.extend(chosen)
            stats[name] = {**sources, "chosen": len(chosen), "checked": dict(reasons)}
            write_manifest(settings, stats, photos, complete=len(stats) == len(people))
            print(f"{name}: {len(chosen)} chosen ({sources}) {dict(reasons)}", flush=True)
    print(f"{len(photos)} photos of {len(stats)} people written to {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
