"""Accuracy on real photos (Wikimedia Commons) through the full /predict path.

Run with: uv run python -m ml.evaluate_real
Downloads any missing images listed in data/real_photos/manifest.json, checks
their SHA-256, and posts each one to the app with face detection on, so
decoding, EXIF rotation, face crop, preprocessing, and the unknown threshold
all match serving. Prints Markdown tables for docs/REAL_PHOTOS.md.
"""

import json
from collections import defaultdict

from fastapi.testclient import TestClient

from app.main import UNKNOWN_IDENTITY, app
from app.schemas import ModelName
from ml.real_photos import MANIFEST_FILE, download_images, image_path, load_manifest

MODELS = ["cnn", "cnn-v2", "pca-knn"]
LFW_ERA_LAST_YEAR = 2005  # LFW photos are news images from 2002-2004


def era(year: int | None) -> str:
    if year is None:
        return "unknown year"
    return f"until {LFW_ERA_LAST_YEAR}" if year <= LFW_ERA_LAST_YEAR else f"after {LFW_ERA_LAST_YEAR}"


def predict_all(client: TestClient, photos: list[dict], model: str) -> list[dict]:
    results = []
    for photo in photos:
        path = image_path(photo)
        response = client.post(
            "/predict", params={"model": model}, files={"file": (path.name, path.read_bytes())}
        )
        if response.status_code == 422 and response.json()["detail"] == "no face detected":
            results.append({"detected": False})
            continue
        response.raise_for_status()
        body = response.json()
        names = [p["name"] for p in body["predictions"]]
        results.append({
            "detected": True,
            "top1": names[0] == photo["label"],
            "top3": photo["label"] in names,
            "accepted": body["identity"] != UNKNOWN_IDENTITY,
        })
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    accepted = [r for r in results if r.get("accepted")]
    return {
        "photos": total,
        "detected": sum(r["detected"] for r in results) / total,
        "top1": sum(r.get("top1", False) for r in results) / total,
        "top3": sum(r.get("top3", False) for r in results) / total,
        "accepted": len(accepted) / total,
        "accepted_correct": sum(r["top1"] for r in accepted),
        "accepted_count": len(accepted),
    }


def per_person_top1(photos: list[dict], results: list[dict]) -> float:
    """Mean of each person's top-1 accuracy, so people with more photos don't dominate."""
    by_person = defaultdict(list)
    for photo, result in zip(photos, results, strict=True):
        by_person[photo["label"]].append(result.get("top1", False))
    return sum(sum(hits) / len(hits) for hits in by_person.values()) / len(by_person)


def grouped(title: str, key, names, photos, by_model) -> None:
    print(f"\n| 모델 | {title} | 사진 수 | top-1 | top-3 |")
    print("|---|---|---|---|---|")
    for model, results in by_model.items():
        groups = defaultdict(list)
        for photo, result in zip(photos, results, strict=True):
            groups[key(photo)].append(result)
        for name in names:
            if groups[name]:
                s = summarize(groups[name])
                print(f"| {model} | {name} | {s['photos']} | {percent(s['top1'])} | {percent(s['top3'])} |")


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    photos = load_manifest()
    if not json.loads(MANIFEST_FILE.read_text(encoding="utf-8")).get("complete"):
        print("warning: the manifest is incomplete; scripts/collect_real_photos.py did not finish\n")
    download_images(photos)

    with TestClient(app) as client:
        lfw_accuracy = {
            model: app.state.models[ModelName(model)].manifest["unknown_calibration"]["known_test"][
                "accuracy"
            ]
            for model in MODELS
        }
        by_model = {model: predict_all(client, photos, model) for model in MODELS}

    print(f"{len(photos)} photos of {len({p['label'] for p in photos})} people\n")
    print("| 모델 | 검출률 | top-1 | top-1 (사람별 평균) | top-3 | 통과율 | 통과한 것 중 정답 | 참고: LFW test top-1 (검출) |")
    print("|---|---|---|---|---|---|---|---|")
    for model, results in by_model.items():
        s = summarize(results)
        precision = (
            f"{s['accepted_correct']}/{s['accepted_count']}" if s["accepted_count"] else "-"
        )
        print(
            f"| {model} | {percent(s['detected'])} | {percent(s['top1'])} "
            f"| {percent(per_person_top1(photos, results))} | {percent(s['top3'])} "
            f"| {percent(s['accepted'])} | {precision} | {percent(lfw_accuracy[model])} |"
        )

    grouped("촬영 시기", lambda p: era(p["year"]),
            (f"until {LFW_ERA_LAST_YEAR}", f"after {LFW_ERA_LAST_YEAR}", "unknown year"),
            photos, by_model)
    grouped("라벨 출처", lambda p: p["label_source"], ("depicts", "category"), photos, by_model)


if __name__ == "__main__":
    main()
