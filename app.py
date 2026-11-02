import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

ROOT = Path(__file__).parent
REFERENCE_DIR = ROOT / "reference_images"
PRODUCTS_FILE = ROOT / "products.json"
MIN_INLIERS = 8
MIN_WINNER_MARGIN = 1.15

app = FastAPI(title="Medicine Box Face Matcher")
orb = cv2.ORB_create(nfeatures=1800)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def load_products() -> dict[str, Any]:
    if not PRODUCTS_FILE.exists():
        return {}
    return json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))


def load_references() -> list[dict[str, Any]]:
    references = []
    for path in sorted(REFERENCE_DIR.glob("*__*__*.jpg")):
        # Naming: product_id__face__sample.jpg
        parts = path.stem.split("__")
        if len(parts) != 3:
            continue
        product_id, face, _sample = parts
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None and len(keypoints) >= 4:
            references.append({
                "product_id": product_id,
                "face": face,
                "path": path.name,
                "keypoints": keypoints,
                "descriptors": descriptors,
            })
    return references


def compare_features(
    query_keypoints: list[cv2.KeyPoint],
    query_descriptors: np.ndarray,
    reference: dict[str, Any],
) -> tuple[int, int]:
    pairs = matcher.knnMatch(query_descriptors, reference["descriptors"], k=2)
    good = [first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        return len(good), 0

    source_points = np.float32(
        [query_keypoints[item.queryIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [reference["keypoints"][item.trainIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    _homography, inlier_mask = cv2.findHomography(
        source_points, target_points, cv2.RANSAC, 5.0
    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return len(good), inliers


def rank_faces(image: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    query_keypoints, query_descriptors = orb.detectAndCompute(gray, None)
    if query_descriptors is None or len(query_keypoints) < 4:
        return []

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for reference in load_references():
        good_matches, inliers = compare_features(
            query_keypoints, query_descriptors, reference
        )
        key = (reference["product_id"], reference["face"])
        candidate = grouped.setdefault(key, {
            "product_id": key[0],
            "face": key[1],
            "good_matches": 0,
            "inliers": 0,
            "reference_image": reference["path"],
        })
        if inliers > candidate["inliers"]:
            candidate.update({
                "good_matches": good_matches,
                "inliers": inliers,
                "reference_image": reference["path"],
            })
    return sorted(grouped.values(), key=lambda item: item["inliers"], reverse=True)


def make_guidance(product: dict[str, Any], face: str, target: str) -> str:
    order = product.get("face_order", [])
    if face not in order or target not in order:
        return "회전 방향 정보를 설정하지 않았습니다."
    if face == target:
        return "필요한 정보가 있는 면을 찾았습니다."

    current_index = order.index(face)
    target_index = order.index(target)
    clockwise_steps = (target_index - current_index) % len(order)
    counter_steps = (current_index - target_index) % len(order)
    direction = "시계방향" if clockwise_steps <= counter_steps else "반시계방향"
    steps = min(clockwise_steps, counter_steps)
    return f"{direction}으로 {steps}면 돌려 주세요. 한 면씩 돌린 뒤 다시 스캔해 주세요."


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scan")
async def scan(
    image: UploadFile = File(...),
    target: str | None = Query(default=None, description="예: expiry, essential, extra"),
) -> dict[str, Any]:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="JPEG 또는 PNG 이미지를 보내 주세요.")
    raw = await image.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="이미지를 읽지 못했습니다.")

    products = load_products()
    ranked = rank_faces(frame)
    if not ranked:
        return {
            "recognized": False,
            "message": "비교할 특징을 찾지 못했습니다. 상자 면을 카메라에 맞춰 다시 촬영해 주세요.",
        }

    best = ranked[0]
    runner_up = ranked[1]["inliers"] if len(ranked) > 1 else 0
    if (best["inliers"] < MIN_INLIERS or
            (runner_up > 0 and best["inliers"] < runner_up * MIN_WINNER_MARGIN)):
        return {
            "recognized": False,
            "message": "촬영 이미지와 등록 이미지가 충분히 일치하지 않습니다. 다시 촬영해 주세요.",
            "top_candidates": ranked[:3],
        }

    product = products.get(best["product_id"])
    if not product:
        return {
            "recognized": False,
            "message": "제품 정보 설정을 찾지 못했습니다.",
            "candidate": best,
        }

    target_face = product.get("targets", {}).get(target) if target else None
    guidance = make_guidance(product, best["face"], target_face) if target_face else None
    return {
        "recognized": True,
        "product_id": best["product_id"],
        "product_name": product.get("name", best["product_id"]),
        "face": best["face"],
        "inlier_matches": best["inliers"],
        "reference_image": best["reference_image"],
        "target_face": target_face,
        "guidance": guidance,
        "message": "제품과 포장 면을 이미지 특징 비교로 식별했습니다.",
    }
