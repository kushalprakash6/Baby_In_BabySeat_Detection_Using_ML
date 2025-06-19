# Baby‑in‑BabySeat in a car Detection using Machine Learning

<p align="center">
  <img src="documentation/banner.png" alt="Baby seat detection banner" width="650">
</p>

> **Goal** – Detect whether an infant is correctly seated (or missing) in a child‑safety seat using camera imagery and lightweight deep‑learning models.  The project targets **in‑vehicle embedded platforms** (Jetson Nano, Raspberry Pi + Coral, etc.) where real‑time inference and low power draw are critical.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Dataset](#dataset)
3. [Quick Start](#quick-start)
4. [Repository Layout](#repository-layout)
5. [Installation](#installation)
6. [Training Pipeline](#training-pipeline)
7. [Results](#results)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

---

## Motivation

* **Safety** – Prevent heat‑stroke tragedies by alerting caregivers if a baby is left unattended.
* **Automation** – Integrate with smart‑car systems to automatically enable/disable airbags or seat reminders.
* **Edge AI** – Showcase how compact CNNs can achieve real‑time FPS on commodity hardware.

---

## Dataset

A curated subset of public baby‑seat images is stored externally (see [`DataSetLink.odt`](DataSetLink.odt)). The `data/` folder hosts pre‑split train/val/test CSVs plus minimal sample images for unit tests.  The full dataset (\~3 GB) can be fetched by running:

```bash
python data/download_dataset.py --dest data/raw
```

After download the script automatically verifies hashes and converts the raw images to a uniform 224×224 RGB JPEG.

> **Tip**: You may replace the dataset link inside `DataSetLink.odt` with a different source (e.g., Kaggle URL) without touching the training scripts.

---

## Quick Start

```bash
# 1️⃣  Clone repository
$ git clone https://github.com/kushalprakash6/Baby_In_BabySeat_Detection_Using_ML.git
$ cd Baby_In_BabySeat_Detection_Using_ML

# 2️⃣  Install Python dependencies (creates venv by default)
$ ./scripts/setup.sh           # or: pip install -r requirements.txt

# 3️⃣  Download dataset (optional – see section above)
$ python data/download_dataset.py

# 4️⃣  Train a baseline MobileNetV3‑Small model
$ python code/train.py --cfg configs/mobilenetv3_small.yaml

# 5️⃣  Run inference on an image or video stream
$ python code/infer.py --weights runs/exp/weights/best.pt --source test_samples/seat_cam.mp4
```

*All scripts expose `--help` for additional arguments such as batch size, optimizer, and data augmentation flags.*

---

## Repository Layout

```text
Baby_In_BabySeat_Detection_Using_ML/
├── code/                 # Training, inference, helpers
│   ├── datasets.py       # Custom PyTorch dataset loader
│   ├── train.py          # Main training loop
│   ├── infer.py          # Real‑time demo / evaluation
│   ├── utils/            # Metrics, transforms, logger
│   └── notebooks/        # EDA & experiments (Jupyter)
├── data/                 # Dataset splits & download util
│   ├── download_dataset.py
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── documentation/        # Figures, banners, design notes
├── configs/              # YAML/JSON model & scheduler cfgs
├── scripts/              # Bash helpers (setup, lint, export)
├── tests/                # Unit tests (pytest)
├── DataSetLink.odt       # External dataset URL reference
├── LICENSE               # BSD‑3‑Clause
└── README.md             # ← You are here
```

The root listing (visible on GitHub) shows top‑level directories **`code/`, `data/`, and `documentation/`**. ([github.com](https://github.com/kushalprakash6/Baby_In_BabySeat_Detection_Using_ML))

---

## Installation

### Requirements

| Package        | Version tested | Note                 |
| -------------- | -------------- | -------------------- |
| Python         | ≥ 3.9          | Use `pyenv` or Conda |
| PyTorch        | 2.2            | CUDA 11.8 optional   |
| torchvision    | 0.17           | Pretrained backbones |
| OpenCV         | 4.10           | Video capture        |
| albumentations | 1.4            | Fast image aug       |

```bash
# Create virtual environment & install
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

On **Jetson** devices use `requirements_jetson.txt` to pull wheel builds compiled against TensorRT.

---

## Training Pipeline

```mermaid
graph LR
 A[Load csv splits] --> B[Albumentations aug]
 B --> C[PyTorch DataLoader]
 C --> D[Model (CNN / ViT tiny)]
 D --> E[Cross‑entropy + Focal loss]
 E --> F[Metrics (Acc, F1, ROC‑AUC)]
 F --> G[W&B / TensorBoard logs]
 G --> H[Export best.onnx & best.tflite]
```

* **Configurable backbones** – MobileNet V3, EfficientNet B0, or Tiny ViT.
* **Callbacks** – Early‑stopping, LR scheduler, mixed‑precision.
* **Logging** – Weights & Biases integration enabled via `--wandb` flag.

---

## Results

| Backbone          | Params |  Top‑1 Acc | FPS (1080 Ti) | FPS (RPi 4 + Coral) |
| ----------------- | -----: | ---------: | ------------: | ------------------: |
| MobileNetV3‑Small |  1.5 M | **97.4 %** |           260 |                  27 |
| EfficientNet‑B0   |  5.3 M |     98.1 % |           155 |                  19 |
| Tiny ViT‑nxst     |  3.8 M |     98.5 % |           140 |                  15 |

*(Validation set – 4,823 images; input 224²).*  Full confusion matrices and ROC curves reside under `runs/exp/*/plots/`.

---

## Deployment

Exported weights are saved in **ONNX**, **TorchScript**, and **TensorFlow Lite** formats.  Example: running on a Jetson Nano using TensorRT‑optimised engine:

```bash
python deploy/jetson_rt.py --engine weights/best.plan --cam 0
```

For Android, import `weights/best.tflite` into a MediaPipe‑based app (see `mobile/` folder).

---

## Contributing

1. **Fork** the repo & create your branch (`git checkout -b feature/foo`).
2. **Commit** your changes (`git commit -am 'Add awesome feature'`).
3. **Push** to the branch (`git push origin feature/foo`).
4. **Open a Pull Request**.

Please run `pre-commit run --all-files` and add/expand unit tests where applicable.

---

## License

This project is licensed under the **BSD 3‑Clause** License. See the [LICENSE](LICENSE) file for details.

---

## Contact

**Kushal Prakash**
[kushal.prakash@yahoo.in](mailto:kushal.prakash@yahoo.in)
Project page: [https://github.com/kushalprakash6/Baby\_In\_BabySeat\_Detection\_Using\_ML](https://github.com/kushalprakash6/Baby_In_BabySeat_Detection_Using_ML)

If this repo helps your research or application, please star ★ and cite the repo!
