import os
import json
import glob
import argparse
from typing import Dict, List, Any


def parse_gt_file(txt_path: str) -> List[Dict[str, Any]]:
    """
    Parse one VinText original annotation file.

    Each line is expected to look like:
    x1,y1,x2,y2,x3,y3,x4,y4,text

    Returns a list of dicts:
    {
        "poly": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
        "text": "...",
        "ignore": bool
    }
    """
    items: List[Dict[str, Any]] = []

    with open(txt_path, "r", encoding="utf-8-sig") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 9:
                print(f"[WARN] Skip malformed line {line_idx} in {txt_path}: {line}")
                continue

            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                print(f"[WARN] Non-integer coords at line {line_idx} in {txt_path}: {line}")
                continue

            # Join the remaining fields back in case transcript contains commas
            text = ",".join(parts[8:]).strip()

            poly = [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]],
            ]

            items.append(
                {
                    "poly": poly,
                    "text": text,
                    "ignore": text == "###",
                }
            )

    return items


def find_image_name(image_dir: str, idx: str) -> str:
    """
    Try to match gt_<idx>.txt to an actual image file in image_dir.
    Supports jpg, jpeg, png, bmp, webp.
    """
    exts = ["jpg", "jpeg", "png", "bmp", "webp"]
    for ext in exts:
        candidate = os.path.join(image_dir, f"{idx}.{ext}")
        if os.path.exists(candidate):
            return f"{idx}.{ext}"
    raise FileNotFoundError(
        f"Could not find image for gt_{idx}.txt inside {image_dir}. "
        f"Expected one of: {', '.join(f'{idx}.{e}' for e in exts)}"
    )


def build_gt_json(labels_dir: str, image_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    gt_dict: Dict[str, List[Dict[str, Any]]] = {}

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )

    for img_name in image_files:

        stem = os.path.splitext(img_name)[0]  # img1201

        # lấy số 1201
        digits = "".join(c for c in stem if c.isdigit())

        label_name = f"gt_{digits}.txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            print(f"[WARN] label not found for {img_name}")
            continue

        gt_dict[img_name] = parse_gt_file(label_path)

    return gt_dict


def summarize(gt_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    num_images = len(gt_dict)
    num_instances = 0
    num_ignore = 0

    for items in gt_dict.values():
        num_instances += len(items)
        num_ignore += sum(1 for x in items if x["ignore"])

    return {
        "num_images": num_images,
        "num_instances": num_instances,
        "num_ignore": num_ignore,
        "num_valid": num_instances - num_ignore,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VinText original gt_*.txt annotations to a single JSON file."
    )
    parser.add_argument("--labels_dir", required=True, help="Path to labels folder containing gt_*.txt")
    parser.add_argument("--image_dir", required=True, help="Path to image folder, e.g. test_image")
    parser.add_argument("--output", required=True, help="Output JSON path, e.g. gt_test.json")
    args = parser.parse_args()

    gt_dict = build_gt_json(args.labels_dir, args.image_dir)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gt_dict, f, ensure_ascii=False, indent=2)

    stats = summarize(gt_dict)
    print(f"[OK] Saved GT JSON to: {args.output}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # Show one sample entry for quick inspection
    first_key = next(iter(gt_dict))
    print("\nSample entry:")
    print(json.dumps({first_key: gt_dict[first_key][:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
