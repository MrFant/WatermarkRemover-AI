---
title: WatermarkRemover-AI
app_file: webui.py
sdk: gradio
sdk_version: 4.21.0
---
# Watermark Remover AI

这是一个功能强大的AI水印去除工具，它利用深度学习模型智能地检测并移除图片和视频中的水印。

## ✨ 主要特性

- **支持多种媒体**：可处理图片（`.png`, `.jpg`等）和视频（`.mp4`, `.mov`等）文件。
- **双检测引擎**：
  - **YOLO模型**：速度快，适用于检测固定样式的水印。
  - **Florence-2模型**：灵活性高，可通过文本提示（如“白色logo”）来定位水印。
- **高质量修复**：使用LaMa模型进行图像修复（Inpainting），效果自然，痕迹少。
- **双操作界面**：
  - **WebUI**：提供简单易用的图形化界面，所有参数均有详细说明，适合大多数用户。
  - **命令行**：通过 `main.py` 提供完整的命令行接口，方便集成和自动化调用。
- **性能优化**：
  - **批量处理**：对检测和修复环节都实现了批量处理，大幅提升视频处理速度。
  - **GPU加速**：完整支持NVIDIA GPU（CUDA），并为视频编码提供NVENC硬件加速支持。
  - **半精度推理**：支持FP16半精度，在兼容的GPU上可减少显存占用并提升速度。

---

## 🚀 安装与设置

### 1. 环境要求

- Python 3.10+
- `pip` (Python包管理器)
- **FFmpeg**：处理视频音频的必要工具。请确保已在你的系统中安装，并将其添加到了系统环境变量（PATH）中。
- **NVIDIA GPU (推荐)**：为了获得理想的处理速度，强烈建议使用NVIDIA显卡，并安装相应的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 和驱动程序。

### 2. 安装步骤

1.  **克隆仓库** (如果还未下载):
    ```bash
    git clone git@github.com:MrFant/WatermarkRemover-AI.git
    cd WatermarkRemover-AI
    ```

2.  **安装Python依赖**:
    我们已经为你准备好了 `requirements.txt` 文件。执行以下命令一键安装所有必需的库：
    ```bash
    pip install -r requirements.txt
    ```

3.  **下载模型文件**:
    本项目需要两个预训练模型文件，请手动下载并将它们放置在正确的目录中：

    - **YOLO 模型** (用于检测):
      - 将你的YOLO模型文件（例如 `yolo.pt`）放置在 `models/` 目录下。
      - 最终路径应为：`models/yolo.pt`

    - **LaMa 模型** (用于修复):
      - 从 [这里](https://github.com/saic-mdal/lama/releases/download/v1.0/big-lama.pt) 下载 `big-lama.pt` 文件。
      - 将下载的文件放置在 `models/` 目录下。
      - 最终路径应为：`models/big-lama.pt`

    *注意：Florence-2模型会在首次使用时由`transformers`库自动下载。*

---

## 🕹️ 使用方法

你可以通过两种方式使用本工具：

### 1. WebUI (推荐)

这是最简单直观的使用方式。在项目根目录下运行：

```bash
python webui.py
```

程序会启动一个本地Web服务（通常地址为 `http://127.0.0.1:7860`）。在浏览器中打开此地址即可看到操作界面。所有参数在界面上都有通俗易懂的中文解释。

### 2. 命令行 (`main.py`)

对于高级用户或需要自动化的场景，可以直接使用 `main.py`。

**基本用法:**
```bash
python main.py --input <输入文件或目录> --output <输出目录> [其他选项]
```

**主要命令行参数详解:**

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--input` | `input` | 输入文件或文件夹的路径。 |
| `--output` | `output` | 输出结果的保存目录。 |
| `--model` | `yolo` | 选择检测模型，可选 `yolo` 或 `florence`。 |
| `--device` | `cuda` 或 `cpu` | 运行设备，默认自动检测。 |
| `--half-precision` / `--no-half-precision` | 启用 | 是否启用FP16半精度推理。 |
| `--video-batch-size` | `8` | 处理视频时的批大小，影响速度和显存占用。 |
| `--use-first-frame-detection` | 禁用 | 是否对视频启用“首帧检测”模式。 |

**YOLO专用参数:**
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--yolo-model` | `models/yolo.pt` | YOLO模型文件的路径。 |
| `--yolo-imgsz` | `640` | YOLO模型推理时使用的图像尺寸。 |
| `--conf-threshold` | `0.6` | YOLO检测的置信度阈值。 |
| `--iou-threshold` | `0.45` | YOLO的IOU阈值。 |

**Florence-2专用参数:**
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--model-size` | `large` | Florence-2模型的大小，可选 `base` 或 `large`。 |
| `--watermark-text` | `"white English text watermark"` | 用于提示Florence-2识别水印的文本。 |

---

## ⚡ 性能优化建议

- **使用GPU**：这是最重要的性能保障。请确保你的环境已正确配置CUDA。
- **首帧检测**：处理水印位置固定的视频时，务必在WebUI中勾选“Use First Frame Mask for Videos”，或在命令行使用 `--use-first-frame-detection`。这是**最有效**的视频处理加速手段。
- **调整批大小 (`Video Batch Size`)**：根据你的GPU显存大小，适当增大此参数可以提升GPU利用率，加快视频处理速度。如果遇到显存不足（Out of Memory）的错误，请减小此值。
- **调整YOLO图像尺寸 (`YOLO Image Size`)**：减小此值（如从1280降到640）可以显著加快YOLO的检测速度，但可能会牺牲对微小水印的检测精度。你可以根据实际情况进行权衡。