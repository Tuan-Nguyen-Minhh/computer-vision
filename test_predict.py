import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from typing import List, Tuple

# =========================
# 1. Các hàm hình học (Geometry)
# =========================
def polygon_iou(poly1: List[List[int]], poly2: List[List[int]]) -> float:
    p1 = np.array(poly1, dtype=np.float32)
    p2 = np.array(poly2, dtype=np.float32)
    all_pts = np.vstack([p1, p2])
    min_x, min_y = int(np.floor(np.min(all_pts[:, 0]))), int(np.floor(np.min(all_pts[:, 1])))
    max_x, max_y = int(np.ceil(np.max(all_pts[:, 0]))), int(np.ceil(np.max(all_pts[:, 1])))
    w, h = max_x - min_x + 3, max_y - min_y + 3
    if w <= 0 or h <= 0: return 0.0
    p1_shift = (p1 - np.array([min_x - 1, min_y - 1], dtype=np.float32)).astype(np.int32)
    p2_shift = (p2 - np.array([min_x - 1, min_y - 1], dtype=np.float32)).astype(np.int32)
    mask1, mask2 = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask1, [p1_shift], 1)
    cv2.fillPoly(mask2, [p2_shift], 1)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union != 0 else 0.0

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def polygon_bbox_size(poly: List[List[int]]) -> Tuple[float, float, float]:
    pts = np.array(poly, dtype=np.float32)
    w = np.max(pts[:, 0]) - np.min(pts[:, 0])
    h = np.max(pts[:, 1]) - np.min(pts[:, 1])
    return w, h, w * h

def warp_polygon_crop(img: np.ndarray, poly: List[List[int]], out_h: int = 64, margin: int = 4) -> np.ndarray:
    pts = order_quad_points(np.array(poly, dtype=np.float32))
    tl, tr, br, bl = pts
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))) + 2 * margin
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))) + 2 * margin
    out_w = max(8, int(max(width, 8) * (out_h / float(max(height, 8)))))
    dst = np.array([[margin, margin], [out_w - 1 - margin, margin], 
                    [out_w - 1 - margin, out_h - 1 - margin], [margin, out_h - 1 - margin]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# =========================
# 2. Tiền xử lý & VietOCR
# =========================
def apply_clahe(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.cvtColor(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), cv2.COLOR_GRAY2BGR)

def apply_gamma(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_sharpen(img: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(img, 1.6, cv2.GaussianBlur(img, (0, 0), 1.2), -0.6, 0)

def preprocess_crop(img: np.ndarray) -> np.ndarray:
    return apply_sharpen(apply_gamma(apply_clahe(img), gamma=1.2))

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
    pil_img = Image.fromarray(img) if img.ndim == 2 else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return " ".join(detector.predict(pil_img).strip().split())

def refine_pred_boxes(pred_polys, pred_confs, conf_thr=0.25, min_area=80.0, nms_iou_thr=0.4):
    candidates = [(p, c) for p, c in zip(pred_polys, pred_confs) if c >= conf_thr and polygon_bbox_size(p)[2] >= min_area]
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept_polys, kept_confs = [], []
    for poly, conf in candidates:
        if not any(polygon_iou(poly, kp) >= nms_iou_thr for kp in kept_polys):
            kept_polys.append(poly)
            kept_confs.append(conf)
    return kept_polys, kept_confs

# =========================
# 3. Vẽ kết quả (sửa lỗi ???)
# =========================
def draw_predictions(img_bgr: np.ndarray, polys: list, texts: list, font_path: str = "arial.ttf") -> np.ndarray:
    # 1. Vẽ khung đa giác bằng OpenCV
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 2. Khởi tạo canvas PIL với chế độ RGBA để hỗ trợ độ trong suốt tốt nhất
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, 13) # Giảm cỡ chữ xuống 18 cho thanh thoát
    except IOError:
        print(f"[CẢNH BÁO] Không tìm thấy font '{font_path}'.")
        font = ImageFont.load_default()

    for poly, text in zip(polys, texts):
        # Bước A: Tìm Bounding Box hình chữ nhật bao quanh đa giác OBB
        pts = np.array(poly, dtype=np.int32)
        x_min, y_min = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
        x_max, y_max = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

        # Bước B: Đo đạc kích thước thực tế của chuỗi Text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Bước C: Thuật toán định vị & Chống tràn (Collision Detection)
        pad = 4 # Vùng đệm 4 pixels xung quanh chữ
        text_x = x_min
        text_y = y_min - text_h - (pad * 2) # Vị trí ưu tiên: Ngay trên nóc box
        
        # Nếu chữ bị tràn lên cạnh trên cùng của ảnh, lật nó xuống dưới đáy box
        if text_y < 0:
            text_y = y_max + 2

        # Bước D: Render với độ tương phản cao (Chữ Trắng - Nền Đen Mờ)
        bg_rect = [text_x, text_y, text_x + text_w + (pad * 2), text_y + text_h + (pad * 2)]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 180)) # 180 là độ mờ (Alpha)
        
        # Đặt chữ vào tâm của vùng nền (cộng thêm padding)
        draw.text((text_x + pad, text_y + pad - 2), text, font=font, fill=(255, 255, 255, 255))

    # Chuyển ngược lại về BGR để OpenCV có thể lưu ảnh
    final_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)

# =========================
# 4. Hàm chạy chính (Main)
# =========================
def predict_single_image(model_path: str, img_path: str, out_dir: str, font_path: str = "arial.ttf"):
    os.makedirs(out_dir, exist_ok=True)
    
    print("[1/3] Đang tải mô hình vào RAM/VRAM...")
    model = YOLO(model_path)
    ocr_model = load_vietocr_predictor("vgg_transformer")
    
    print(f"[2/3] Đang đọc ảnh: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[LỖI] Không thể đọc được ảnh tại: {img_path}. Hãy kiểm tra lại đường dẫn!")
        return

    print("      -> Chạy YOLO Detection...")
    result = model(img, conf=0.001, verbose=False)[0]
    raw_polys, raw_confs = [], []
    
    if result.obb is not None and result.obb.xyxyxyxy is not None:
        polys = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        for poly, conf in zip(polys, confs):
            raw_polys.append([[int(x), int(y)] for x, y in poly])
            raw_confs.append(float(conf))

    refined_polys, refined_confs = refine_pred_boxes(raw_polys, raw_confs)
    print(f"      -> Tìm thấy {len(refined_polys)} vùng chứa chữ sau khi lọc (NMS).")
    
    print("      -> Chạy VietOCR (Crop & Text Recognition)...")
    detected_texts = []
    for poly in refined_polys:
        crop = warp_polygon_crop(img, poly, out_h=64, margin=4)
        crop_prep = preprocess_crop(crop)
        text = run_ocr_vietocr(crop_prep, ocr_model)
        detected_texts.append(text)

    print("[3/3] Đang vẽ kết quả và lưu file...")
    vis_img = draw_predictions(img.copy(), refined_polys, detected_texts, font_path)
    
    img_name = os.path.basename(img_path)
    save_path = os.path.join(out_dir, f"result_{img_name}")
    cv2.imwrite(save_path, vis_img)
    print(f"\n[THÀNH CÔNG] Đã lưu ảnh kết quả tại: {save_path}")

if __name__ == "__main__":
    # Đã cấu hình sẵn đường dẫn từ máy Debian của bạn:
    MODEL_PATH = "/home/tuan/Desktop/computer-vision/models/best.pt"
    IMAGE_PATH = "/home/tuan/Desktop/computer-vision/datasets/test_image/im1493.jpg"
    OUTPUT_DIR = "/home/tuan/Desktop/computer-vision/test_outputs"
    FONT_PATH = "arial.ttf" # Bạn nhớ tải 1 file arial.ttf bỏ vào cùng thư mục chạy code nhé

    predict_single_image(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR, FONT_PATH)