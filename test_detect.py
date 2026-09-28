"""Tes deteksi bubble: baca gambar di folder input, gambar kotak hasil, simpan ke folder output.

Pakai:
    python test_detect.py [folder_input] [folder_output] [path_model]
Default:
    /sdcard/rgambar/input  /sdcard/rgambar/output  ~/bubbles_detect.onnx
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
MODEL = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.home() / 'bubbles_detect.onnx'

SIZE = 1024      # ukuran input model
CONF = 0.25      # batas skor minimum
IOU = 0.5        # batas tumpang tindih untuk NMS
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
    return canvas, r, px, py


def detect_tile(tile):
    """Kembalikan list (x1, y1, x2, y2, skor) dalam koordinat tile."""
    th, tw = tile.shape[:2]
    canvas, r, px, py = letterbox(tile)
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {INP: x})[0][0]           # (5, N)
    if out.shape[0] != 5:
        raise RuntimeError(f'bentuk output tak terduga: {out.shape}')
    out = out.T[out[4] >= CONF]                    # (M, 5)
    res = []
    for cx, cy, w, h, s in out:
        x1 = (cx - w / 2 - px) / r
        y1 = (cy - h / 2 - py) / r
        x2 = (cx + w / 2 - px) / r
        y2 = (cy + h / 2 - py) / r
        x1, x2 = max(0.0, x1), min(float(tw), x2)
        y1, y2 = max(0.0, y1), min(float(th), y2)
        if x2 > x1 and y2 > y1:
            res.append((x1, y1, x2, y2, float(s)))
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
    boxes = []
    for y1, y2 in tiles(h, w):
        for bx1, by1, bx2, by2, s in detect_tile(img[y1:y2]):
            boxes.append((bx1, by1 + y1, bx2, by2 + y1, s))
    if not boxes:
        return []
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    scores = [b[4] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores, CONF, IOU)
    return [boxes[int(i)] for i in np.array(keep).flatten()]


def main():
    files = sorted(p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS) if IN_DIR.exists() else []
    if not files:
        print(f'Tidak ada gambar di {IN_DIR}')
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in files:
        t0 = time.time()
        pil = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        img = np.array(pil)
        h, w = img.shape[:2]
        boxes = detect_page(img)

        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        thick = max(2, w // 400)
        for x1, y1, x2, y2, s in boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thick)
            cv2.putText(vis, f'{s:.2f}', (int(x1), max(int(y1) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1500), (0, 0, 255), thick)
        cv2.imwrite(str(OUT_DIR / f'{p.stem}_det.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        data = {
            'image': p.name,
            'size': [w, h],
            'boxes': [dict(x1=round(b[0]), y1=round(b[1]), x2=round(b[2]), y2=round(b[3]),
                           score=round(b[4], 3)) for b in boxes],
        }
        (OUT_DIR / f'{p.stem}_det.json').write_text(json.dumps(data, indent=2))
        print(f'{p.name}: {w}x{h}, {len(boxes)} bubble, {time.time() - t0:.1f} detik')


if __name__ == '__main__':
    main()
