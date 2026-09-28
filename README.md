# pt-to-onnx-converter

Konversi model YOLO `.pt` ke `.onnx` (default: `PSImera/manga_bubbles_detect`, deteksi speech bubble manga). Model diunduh dari Hugging Face, lalu diekspor dengan ultralytics.

## Isi

- `convert_kaggle.ipynb`: notebook mandiri untuk Kaggle.
- `convert.py`: skrip yang sama untuk Colab, laptop, atau Debian proot.

## Kaggle

1. Kaggle > Create > New Notebook > File > Import Notebook > GitHub, pilih repo ini.
2. Settings > Internet: On (butuh verifikasi nomor HP).
3. Run All.
4. Unduh `bubbles_detect.onnx` dari tab Output.

## Colab / lokal

    pip install ultralytics onnx onnxruntime huggingface_hub
    python convert.py

Hasil ada di folder `out/`. Opsi lain: `--repo`, `--file`, `--imgsz` (default 1024), `--out`.

## Catatan

- Ukuran input ONNX tetap sesuai `--imgsz`. Untuk webtoon panjang, potong gambar dulu sebelum deteksi.
- Lisensi model PSImera: MIT. Dataset-nya CC-BY-NC 4.0. Library ultralytics: AGPL-3.0.
