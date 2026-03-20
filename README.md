# Computer Vision Scene Text Project

This repository contains a scene text pipeline for:
- Detecting text regions with YOLO OBB
- Recognizing text with VietOCR
- Evaluating detection and recognition quality across multiple experiment phases

The project includes scripts to build ground-truth JSON, run phased evaluations, and run single-image prediction with visualization.

## Requirements

Recommended Python version: 3.10+

Install dependencies:

```bash
pip install ultralytics opencv-python numpy pillow vietocr torch
```

Notes:
- `torch` can require a CUDA-specific install command depending on your GPU.
- Some scripts default to `Roboto-Regular.ttf` for label rendering. This repo already has `arial.ttf`, so you can pass `--font arial.ttf`.

## Data Format

Each label file under `datasets/labels/` follows:

```text
x1,y1,x2,y2,x3,y3,x4,y4,text
```

## 1) Build Ground Truth JSON

Convert `gt_*.txt` labels into one JSON file:

```bash
python build_gt.py \
	--labels_dir datasets/labels \
	--image_dir datasets/test_image \
	--output datasets/gt_test.json
```

## 2) Run Evaluation - Base

```bash
python evaluation_for_base.py \
	--model models/best.pt \
	--image_dir datasets/test_image \
	--gt_json datasets/gt_test.json \
	--save_debug_dir results/base \
	--font arial.ttf
```

## 3) Run Evaluation - Phase 2 (Box Refinement)

Adds geometric filtering and polygon NMS.

```bash
python evaluation_for_phase2.py \
	--model models/best.pt \
	--image_dir datasets/test_image \
	--gt_json datasets/gt_test.json \
	--save_debug_dir results/phase2 \
	--font arial.ttf
```

## 4) Run Evaluation - Phase 3 (Crop Mode Ablation)

Compares crop modes:
- `warp`: perspective rectification
- `axis`: axis-aligned crop

```bash
python evaluation_for_phase3.py \
	--model models/best.pt \
	--image_dir datasets/test_image \
	--gt_json datasets/gt_test.json \
	--crop_mode warp \
	--save_debug_dir results/phase3 \
	--font arial.ttf
```

## 5) Run Evaluation - Phase 4 (Preprocessing)

Adds OCR pre-processing options:
- `none`
- `clahe`
- `gamma`
- `sharpen`
- `clahe_sharpen`
- `clahe_gamma_sharpen` (default)

```bash
python evaluation_for_phase4.py \
	--model models/best.pt \
	--image_dir datasets/test_image \
	--gt_json datasets/gt_test.json \
	--crop_mode warp \
	--preprocess_mode clahe_gamma_sharpen \
	--save_debug_dir results/phase4 \
	--font arial.ttf
```

## 6) Single Image Prediction

`test_predict.py` runs detection + OCR and writes a rendered result image.

Important:
- The script currently uses hardcoded paths in `__main__`.
- Update `MODEL_PATH`, `IMAGE_PATH`, `OUTPUT_DIR`, and `FONT_PATH` before running.

Then run:

```bash
python test_predict.py
```

## Output

Evaluation scripts print aggregate metrics to console (detection/recognition related), and when `--save_debug_dir` is set they also save:
- Debug visualizations
- Per-image JSON details

## Team members
- [Dang Trung Hieu](https://github.com/handt0311)
- Nguyen Thi Ngoc Lan
- [Nguyen Minh Tuan](https://github.com/Tuan-Nguyen-Minhh)
- [Nguyen Tuan Dung](https://github.com/tuandung1625)