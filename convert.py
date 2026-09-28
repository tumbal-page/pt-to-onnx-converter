"""Download a YOLO .pt from Hugging Face and export it to ONNX.

Usage:
    python convert.py
    python convert.py --repo PSImera/manga_bubbles_detect --file bubbles_detect.pt --imgsz 1024 --out out
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='PSImera/manga_bubbles_detect')
    p.add_argument('--file', default='bubbles_detect.pt')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--out', default='out')
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pt_local = out / Path(a.file).name
    shutil.copy(hf_hub_download(repo_id=a.repo, filename=a.file), pt_local)

    model = YOLO(str(pt_local))
    onnx_path = model.export(format='onnx', imgsz=a.imgsz, simplify=True)
    print('ONNX:', onnx_path)

    # quick sanity check: load with onnxruntime and print I/O shapes
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    for i in s.get_inputs():
        print('in ', i.name, i.shape, i.type)
    for o in s.get_outputs():
        print('out', o.name, o.shape, o.type)
    print('classes:', model.names)


if __name__ == '__main__':
    main()
