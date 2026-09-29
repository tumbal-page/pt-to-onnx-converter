"""Deteksi bubble dengan model SEGMENTASI (kitsumed/yolov8m_seg-speech-bubble, ONNX).

Hasil: bentuk bubble (poligon), bukan kotak.

Pakai:
    python seg_detect.py [folder_input] [folder_output] [path_model]
Default:
    /sdcard/rgambar/input  /sdcard/rgambar/output  ~/model_dynamic.onnx

Output per gambar:
    <nama>_seg.jpg   gambar dengan bentuk bubble berwarna
    <nama>_mask.png  mask hitam-putih (putih = area bubble)
    <nama>_seg.json  poligon, kotak, dan skor tiap bubble
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/sdcard/rgambar/input')
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/sdcard/rgambar/output')
MODEL = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.home() / 'model_dynamic.onnx'

SIZE = 640        # model ini dilatih di 640
CONF = 0.25       # batas skor minimum
IOU = 0.5         # batas tumpang tindih untuk NMS
MASK_THR = 0.5    # ambang mask
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

sess = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
INP = sess.get_inputs()[0].name


def letterbox(img):
    h, w = img.shape[:2]
    r = min(SIZE / w, SIZE / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, px, py, nw, nh


def nms(boxes, scores):
    """boxes: list (x1, y1, x2, y2). Kembalikan indeks yang dipertahankan."""
    if not boxes:
        return []
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], CONF, IOU)
    return [int(i) for i in np.array(keep).flatten()]


def detect_tile(tile):
    """Kembalikan list (box, skor, poligon) dalam koordinat tile."""
    th, tw = tile.shape[:2]
    canvas, r, px, py, nw, nh = letterbox(tile)
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    outs = sess.run(None, {INP: x})
    proto = next(o for o in outs if o.ndim == 4)        # (1, nm, ph, pw)
    pred = next(o for o in outs if o.ndim == 3)[0].T    # (N, 4 + 1 + nm)
    nm, ph, pw = proto.shape[1], proto.shape[2], proto.shape[3]
    if pred.shape[1] != 5 + nm:
        raise RuntimeError(f'bentuk output tak terduga: {pred.shape}, proto {proto.shape}')

    pred = pred[pred[:, 4] >= CONF]
    if len(pred) == 0:
        return []

    lb_boxes = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h in pred[:, :4]]
    keep = nms(lb_boxes, pred[:, 4])

    proto2d = proto[0].reshape(nm, -1)
    # bagian proto tanpa padding letterbox
    c0, c1 = int(round(px * pw / SIZE)), int(round((px + nw) * pw / SIZE))
    r0, r1 = int(round(py * ph / SIZE)), int(round((py + nh) * ph / SIZE))

    res = []
    for i in keep:
        s = float(pred[i, 4])
        coef = pred[i, 5:5 + nm]
        m = 1.0 / (1.0 + np.exp(-(coef @ proto2d)))
        m = m.reshape(ph, pw)[r0:r1, c0:c1]
        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_LINEAR)

        bx1, by1, bx2, by2 = lb_boxes[i]
        x1 = max(0.0, (bx1 - px) / r)
        y1 = max(0.0, (by1 - py) / r)
        x2 = min(float(tw), (bx2 - px) / r)
        y2 = min(float(th), (by2 - py) / r)
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        binmask = np.zeros((th, tw), np.uint8)
        binmask[iy1:iy2, ix1:ix2] = (m[iy1:iy2, ix1:ix2] > MASK_THR).astype(np.uint8)
        contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        res.append(((x1, y1, x2, y2), s, poly))
    return res


def tiles(h, w):
    """Halaman biasa diproses utuh. Gambar sangat tinggi (webtoon) dipotong tumpang tindih."""
    if h <= 1.6 * w:
        yield 0, h
        return
    th = int(1.5 * w)
    step = int(th * 0.85)
    y = 0
    while True:
        y2 = min(y + th, h)
        yield y, y2
        if y2 >= h:
            break
        y += step


def detect_page(img):
    h, w = img.shape[:2]
    items = []
    for y1, y2 in tiles(h, w):
        for (bx1, by1, bx2, by2), s, poly in detect_tile(img[y1:y2]):
            items.append(((bx1, by1 + y1, bx2, by2 + y1), s, poly + np.array([0, y1])))
    keep = nms([it[0] for it in items], [it[1] for it in items])
    return [items[i] for i in keep]


def main():
    files = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS) if IN_DIR.exists() else []
    if not files:
        print(f'Tidak ada gambar di {IN_DIR}')
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in files:
        t0 = time.time()
        img = np.array(ImageOps.exif_transpose(Image.open(p)).convert('RGB'))
        h, w = img.shape[:2]
        items = detect_page(img)

        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        overlay = vis.copy()
        mask = np.zeros((h, w), np.uint8)
        thick = max(2, w // 400)
        polys = [it[2].astype(np.int32) for it in items]
        for poly in polys:
            cv2.fillPoly(overlay, [poly], (0, 255, 0))
            cv2.fillPoly(mask, [poly], 255)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        for poly in polys:
            cv2.polylines(vis, [poly], True, (0, 0, 255), thick)

        cv2.imwrite(str(OUT_DIR / f'{p.stem}_seg.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(OUT_DIR / f'{p.stem}_mask.png'), mask)
        data = {
            'image': p.name,
            'size': [w, h],
            'bubbles': [
                {'score': round(s, 3),
                 'box': [round(b[0]), round(b[1]), round(b[2]), round(b[3])],
                 'polygon': poly.tolist()}
                for (b, s, _), poly in zip(items, polys)
            ],
        }
        (OUT_DIR / f'{p.stem}_seg.json').write_text(json.dumps(data))
        print(f'{p.name}: {w}x{h}, {len(items)} bubble, {time.time() - t0:.1f} detik')


if __name__ == '__main__':
    main()
