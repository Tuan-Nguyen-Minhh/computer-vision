import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


# =========================
# Text utilities
# =========================
def normalize_text(s: str) -> str:
    s = s.strip()
    s = " ".join(s.split())
    return s


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[m]


def cer(pred: str, gt: str) -> float:
    pred = normalize_text(pred)
    gt = normalize_text(gt)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, gt) / max(1, len(gt))


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_text(pred) == normalize_text(gt))


# =========================
# Geometry
# =========================
def polygon_iou(poly1: List[List[int]], poly2: List[List[int]]) -> float:
    p1 = np.array(poly1, dtype=np.float32)
    p2 = np.array(poly2, dtype=np.float32)

    all_pts = np.vstack([p1, p2])
    min_x = int(np.floor(np.min(all_pts[:, 0])))
    min_y = int(np.floor(np.min(all_pts[:, 1])))
    max_x = int(np.ceil(np.max(all_pts[:, 0])))
    max_y = int(np.ceil(np.max(all_pts[:, 1])))

    w = max_x - min_x + 3
    h = max_y - min_y + 3
    if w <= 0 or h <= 0:
        return 0.0

    p1_shift = (p1 - np.array([min_x - 1, min_y - 1], dtype=np.float32)).astype(np.int32)
    p2_shift = (p2 - np.array([min_x - 1, min_y - 1], dtype=np.float32)).astype(np.int32)

    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask1, [p1_shift], 1)
    cv2.fillPoly(mask2, [p2_shift], 1)

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0
    return float(inter) / float(union)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_polygon_crop(
    img: np.ndarray,
    poly: List[List[int]],
    out_h: int = 64,
    margin: int = 4
) -> np.ndarray:
    pts = np.array(poly, dtype=np.float32)
    pts = order_quad_points(pts)

    tl, tr, br, bl = pts
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    width = int(max(w1, w2)) + 2 * margin
    width = max(width, 8)

    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    height = int(max(h1, h2)) + 2 * margin
    height = max(height, 8)

    scale = out_h / float(height)
    out_w = max(8, int(width * scale))

    dst = np.array([
        [margin, margin],
        [out_w - 1 - margin, margin],
        [out_w - 1 - margin, out_h - 1 - margin],
        [margin, out_h - 1 - margin]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(
        img,
        M,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def polygon_bbox_size(poly: List[List[int]]) -> Tuple[float, float, float]:
    pts = np.array(poly, dtype=np.float32)
    min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
    max_x, max_y = np.max(pts[:, 0]), np.max(pts[:, 1])
    w = max_x - min_x
    h = max_y - min_y
    area = w * h
    return w, h, area


# =========================
# VietOCR
# =========================
def load_vietocr_predictor(model_name: str = "vgg_transformer"):
    config = Cfg.load_config_from_name(model_name)
    config["cnn"]["pretrained"] = True
    config["predictor"]["beamsearch"] = True

    try:
        import torch
        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        config["device"] = "cpu"

    return Predictor(config)


def run_ocr_vietocr(img: np.ndarray, detector) -> str:
    if img.ndim == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    text = detector.predict(pil_img)
    return normalize_text(text)


# =========================
# Box post-process (Phase 2)
# =========================
def refine_pred_boxes(
    pred_polys: List[List[List[int]]],
    pred_confs: List[float],
    conf_thr: float = 0.25,
    min_w: float = 8.0,
    min_h: float = 8.0,
    min_area: float = 80.0,
    nms_iou_thr: float = 0.3,
) -> Tuple[List[List[List[int]]], List[float]]:
    candidates = []
    for poly, conf in zip(pred_polys, pred_confs):
        if conf < conf_thr:
            continue

        w, h, area = polygon_bbox_size(poly)
        if w < min_w or h < min_h or area < min_area:
            continue

        candidates.append((poly, conf))

    candidates.sort(key=lambda x: x[1], reverse=True)

    kept_polys = []
    kept_confs = []

    for poly, conf in candidates:
        keep = True
        for kept_poly in kept_polys:
            if polygon_iou(poly, kept_poly) >= nms_iou_thr:
                keep = False
                break
        if keep:
            kept_polys.append(poly)
            kept_confs.append(conf)

    return kept_polys, kept_confs


# =========================
# Matching
# =========================
def greedy_match_predictions_to_gt(
    pred_polys: List[List[List[int]]],
    gt_items: List[Dict[str, Any]],
    iou_thr: float = 0.5
) -> List[Tuple[int, int, float]]:
    candidates = []
    for pi, p in enumerate(pred_polys):
        for gi, gt in enumerate(gt_items):
            if gt.get("ignore", False):
                continue
            iou = polygon_iou(p, gt["poly"])
            if iou >= iou_thr:
                candidates.append((pi, gi, iou))

    candidates.sort(key=lambda x: x[2], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matches = []

    for pi, gi, iou in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, iou))

    return matches


# =========================
# Visualization Khắc phục lỗi ??? 
# =========================
def draw_evaluation_labels(img_bgr: np.ndarray, labels_info: list, font_path: str = "Roboto-Regular.ttf") -> np.ndarray:
    if not labels_info:
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, 14) 
    except IOError:
        print(f"[CẢNH BÁO] Không tìm thấy font '{font_path}'.")
        font = ImageFont.load_default()

    for poly, text in labels_info:
        pts = np.array(poly, dtype=np.int32)
        x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
        x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad = 4
        text_x = x_min
        text_y = y_min - text_h - (pad * 2)

        if text_y < 0:
            text_y = y_max + 2

        bg_rect = [text_x, text_y, text_x + text_w + (pad * 2), text_y + text_h + (pad * 2)]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
        draw.text((text_x + pad, text_y + pad - 2), text, font=font, fill=(255, 255, 255, 255))

    final_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)


# =========================
# Main evaluation
# =========================
def evaluate(
    model_path: str,
    image_dir: str,
    gt_json_path: str,
    conf_thr: float = 0.25,
    match_iou_thr: float = 0.5,
    vietocr_config: str = "vgg_transformer",
    crop_margin: int = 4,
    crop_height: int = 64,
    min_w: float = 8.0,
    min_h: float = 8.0,
    min_area: float = 80.0,
    nms_iou_thr: float = 0.3,
    save_debug_dir: str = None,
    font_path: str = "Roboto-Regular.ttf"
) -> None:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    model = YOLO(model_path)
    ocr_model = load_vietocr_predictor(vietocr_config)

    image_names = sorted(gt_data.keys())

    total_matches = 0
    total_pred_before = 0
    total_pred_after = 0
    total_gt_valid = 0

    cer_list = []
    em_list = []

    per_image_results = []

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    for img_name in image_names:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        gt_items = gt_data[img_name]
        valid_gt_items = [x for x in gt_items if not x.get("ignore", False)]
        total_gt_valid += len(valid_gt_items)

        result = model(img_path, conf=0.001, verbose=False)[0]

        pred_polys = []
        pred_confs = []

        if result.obb is not None and result.obb.xyxyxyxy is not None:
            polys = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()

            for poly, conf in zip(polys, confs):
                poly_int = [[int(x), int(y)] for x, y in poly]
                pred_polys.append(poly_int)
                pred_confs.append(float(conf))

        total_pred_before += len(pred_polys)

        pred_polys, pred_confs = refine_pred_boxes(
            pred_polys=pred_polys,
            pred_confs=pred_confs,
            conf_thr=conf_thr,
            min_w=min_w,
            min_h=min_h,
            min_area=min_area,
            nms_iou_thr=nms_iou_thr,
        )

        total_pred_after += len(pred_polys)

        matches = greedy_match_predictions_to_gt(
            pred_polys,
            gt_items,
            iou_thr=match_iou_thr
        )
        total_matches += len(matches)

        image_cers = []
        image_ems = []

        debug_vis = img.copy() if save_debug_dir else None
        labels_to_draw = [] # Tập hợp nhãn để vẽ bằng PIL

        for match_idx, (pi, gi, iou) in enumerate(matches):
            pred_poly = pred_polys[pi]
            gt_item = gt_items[gi]

            crop = warp_polygon_crop(
                img,
                pred_poly,
                out_h=crop_height,
                margin=crop_margin
            )
            pred_text = run_ocr_vietocr(crop, ocr_model)
            gt_text = normalize_text(gt_item["text"])

            c = cer(pred_text, gt_text)
            e = exact_match(pred_text, gt_text)

            cer_list.append(c)
            em_list.append(e)
            image_cers.append(c)
            image_ems.append(e)

            if debug_vis is not None:
                pred_contour = np.array(pred_poly, dtype=np.int32).reshape(-1, 1, 2)
                gt_contour = np.array(gt_item["poly"], dtype=np.int32).reshape(-1, 1, 2)

                cv2.polylines(debug_vis, [pred_contour], True, (0, 255, 0), 2)   # pred xanh lá
                cv2.polylines(debug_vis, [gt_contour], True, (255, 0, 0), 2)     # gt xanh dương

                # Chỉ in chữ dự đoán để ảnh gọn gàng hơn
                label = pred_text
                labels_to_draw.append((pred_poly, label))

                crop_path = os.path.join(
                    save_debug_dir,
                    f"{os.path.splitext(img_name)[0]}_match{match_idx}_crop.png"
                )
                cv2.imwrite(crop_path, crop)

        # Vẽ toàn bộ text bằng PIL
        if debug_vis is not None:
            debug_vis = draw_evaluation_labels(debug_vis, labels_to_draw, font_path)
            vis_path = os.path.join(save_debug_dir, f"{os.path.splitext(img_name)[0]}_vis.jpg")
            cv2.imwrite(vis_path, debug_vis)

        per_image_results.append({
            "image": img_name,
            "num_pred_before_refine": int(len(pred_confs) + (len(pred_polys) - len(pred_confs)) if False else 0),
            "num_pred_after_refine": len(pred_polys),
            "num_gt_valid": len(valid_gt_items),
            "num_matches": len(matches),
            "cer_mean": float(np.mean(image_cers)) if image_cers else None,
            "exact_match_rate": float(np.mean(image_ems)) if image_ems else None,
        })

        per_image_results[-1]["num_pred_before_refine"] = int(
            len(result.obb.xyxyxyxy) if (result.obb is not None and result.obb.xyxyxyxy is not None) else 0
        )

        print(
            f"[{img_name}] pred_before={per_image_results[-1]['num_pred_before_refine']} "
            f"| pred_after={len(pred_polys)} | gt_valid={len(valid_gt_items)} "
            f"| matched={len(matches)} | CER={np.mean(image_cers) if image_cers else 'NA'} "
            f"| EM={np.mean(image_ems) if image_ems else 'NA'}"
        )

    summary = {
        "num_images": len(per_image_results),
        "total_pred_boxes_before_refine": total_pred_before,
        "total_pred_boxes_after_refine": total_pred_after,
        "total_valid_gt_boxes": total_gt_valid,
        "total_matched_boxes": total_matches,
        "match_rate_over_pred": total_matches / total_pred_after if total_pred_after > 0 else 0.0,
        "match_rate_over_gt": total_matches / total_gt_valid if total_gt_valid > 0 else 0.0,
        "CER_mean": float(np.mean(cer_list)) if cer_list else None,
        "ExactMatch_rate": float(np.mean(em_list)) if em_list else None,
        "vietocr_config": vietocr_config,
        "crop_margin": crop_margin,
        "crop_height": crop_height,
        "preprocess": False,
        "phase": "phase2_box_postprocess",
        "box_postprocess": {
            "conf_thr": conf_thr,
            "min_w": min_w,
            "min_h": min_h,
            "min_area": min_area,
            "nms_iou_thr": nms_iou_thr,
        },
        "match_iou_thr": match_iou_thr,
    }

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if save_debug_dir:
        summary_path = os.path.join(save_debug_dir, "summary.json")
        detail_path = os.path.join(save_debug_dir, "per_image_results.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(per_image_results, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved debug outputs to: {save_debug_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: YOLOv8-OBB -> box post-process -> VietOCR -> instance-level evaluation"
    )
    parser.add_argument("--model", required=True, help="Path to YOLOv8 OBB weights")
    parser.add_argument("--image_dir", required=True, help="Path to image folder")
    parser.add_argument("--gt_json", required=True, help="Path to gt_test.json")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for box refinement")
    parser.add_argument("--match_iou_thr", type=float, default=0.5, help="IoU threshold for pred-GT matching")

    parser.add_argument("--vietocr_config", default="vgg_transformer",
                        help="VietOCR config name, e.g. vgg_transformer, vgg_seq2seq")
    parser.add_argument("--crop_margin", type=int, default=4, help="Extra margin around warped crop")
    parser.add_argument("--crop_height", type=int, default=64, help="Output crop height after warp")

    parser.add_argument("--min_w", type=float, default=8.0, help="Min bbox width for predicted box")
    parser.add_argument("--min_h", type=float, default=8.0, help="Min bbox height for predicted box")
    parser.add_argument("--min_area", type=float, default=80.0, help="Min bbox area for predicted box")
    parser.add_argument("--nms_iou_thr", type=float, default=0.3, help="Polygon NMS IoU threshold")

    parser.add_argument("--save_debug_dir", default=None, help="Folder to save debug visualizations")
    parser.add_argument("--font", default="Roboto-Regular.ttf", help="Font file path (.ttf) for rendering Vietnamese text")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        image_dir=args.image_dir,
        gt_json_path=args.gt_json,
        conf_thr=args.conf,
        match_iou_thr=args.match_iou_thr,
        vietocr_config=args.vietocr_config,
        crop_margin=args.crop_margin,
        crop_height=args.crop_height,
        min_w=args.min_w,
        min_h=args.min_h,
        min_area=args.min_area,
        nms_iou_thr=args.nms_iou_thr,
        save_debug_dir=args.save_debug_dir,
        font_path=args.font
    )


if __name__ == "__main__":
    main()