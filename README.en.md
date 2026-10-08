# LabelSystem

**A desktop annotation and training assistant for visual defect detection.**

[中文文档](README.md) · [User guide (Chinese)](docs/usage.md) · [AI assistant](docs/ai-assistant.md)

LabelSystem combines rectangle/polygon annotations, sample states, YOLO dataset import/export, cropping, local training/inference and an AI assistant that calls project tools.

![Annotation workspace with synthetic demo images](docs/assets/annotation.png)

## Intended use and annotation format

The main use case is offline industrial visual-defect annotation, sample curation and YOLO model iteration: rectangle annotations for detection, polygons for segmentation, good/defective/false-reject sample management, and local training-result review. Users supply their own images, categories and weights. Real-time camera capture, production-line device control, online multi-user collaboration and a standalone pixel-mask editor are not implemented.

Formal annotations are stored as **one JSON per image**, at `jsons/<image-stem>.json`, with `image_width`, `image_height` and `annotations`. Shapes use `type` (`rect` or `polygon`), the historical category field `lable`, and `points` containing normalized `x`/`y` coordinates. Import supports this structure and some legacy `label`/`category` fields; pixel coordinates can be normalized when valid dimensions are provided. LabelMe/COCO JSON requires conversion first. See the [format and example (Chinese)](docs/data-format.md).

Training exports use YOLO images, TXT labels and `data.yaml`; YOLO detection/segmentation datasets can also be imported into native JSON. Sample lists, statuses and subcategories use separate local management files.

## Published files and privacy

This repository contains source code, documentation, tests and synthetic illustrations. **Our real business images, annotation JSON, user projects, customer information, model weights, training outputs, databases, conversation history and API keys are excluded.** Screenshots show generated data. Runtime private files are ignored by Git; check your own staged files before sharing a project. Bug reports should use synthetic or redacted examples. Local annotation works offline; enabled AI features send their inputs to the provider you configure, as described below.

## Getting started

Use 64-bit Python 3.11 or 3.12. Windows PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe examples/create_demo_project.py
.\.venv\Scripts\python.exe main.py
```

Choose **选择工程文件夹** and open `examples/demo_project`. The generated four-image project contains no real inspection data. For GPU use, install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) before other requirements. Linux/macOS use `.venv/bin/python`; desktop behavior is primarily validated on Windows.

## Features

- Rectangle and polygon annotations, labels, sample states and subcategories.
- JSON annotation and YOLO dataset import; detection/segmentation dataset export.
- Natural-language filtering, statistics, cropping, tiling and export.
- Local `.pt` model training/inference and separately stored prediction candidates.
- Training-result analysis, conversations, shared project resources and human feedback.
- Selected-image analysis, issue sets, previewed annotation/state changes and undo.

Open **AI 助手 → 设置** to supply an API key, complete Chat Completions endpoint and model name. The defaults target a Qwen-compatible service. Vision tasks require image input support. Each selected-image batch contains at most ten source photos; larger selections are paginated. API requests may incur charges and transmit selected images, project summaries and conversations to your configured provider.

No API keys, business datasets, model weights or user history are included. The current release distributes source code, not a verified executable. AI recommendations need human review. Legacy automatic annotation can write formal JSON; candidate predictions remain separate until adoption. Regular inference batches are not yet uniformly persisted. See the Chinese documentation for detailed limitations and data formats.

## Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic fixtures and mocked API calls. Contributing instructions: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU AGPL v3.0: [LICENSE](LICENSE). Third-party components retain their respective licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
