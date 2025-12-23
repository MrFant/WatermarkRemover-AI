# WatermarkRemover-AI 项目说明

## 项目概述

WatermarkRemover-AI 是一个基于深度学习的智能水印去除工具，支持图片和视频中的水印检测与移除。该项目使用 YOLO 或 Florence-2 模型进行水印检测，并利用 LaMa 模型进行高质量图像修复。

## 核心功能

- **双检测引擎**：
  - YOLO 模型：速度快，适用于固定样式水印检测
  - Florence-2 模型：灵活性高，可通过文本提示定位水印
- **高质量修复**：使用 LaMa 模型进行图像修复，效果自然
- **双操作界面**：
  - WebUI：图形化界面，适合大多数用户
  - 命令行：通过 main.py 提供完整的命令行接口
- **性能优化**：
  - 批量处理支持
  - GPU 加速（CUDA）
  - 半精度推理（FP16）

## 项目结构

```
WatermarkRemover-AI/
├── main.py                    # 主程序，命令行入口
├── webui.py                   # WebUI 界面
├── step1_detect_watermark_yolo.py  # YOLO 水印检测脚本
├── requirements.txt           # Python 依赖
├── models/                    # 模型文件目录
│   ├── yolo.pt               # YOLO 检测模型（需手动下载）
│   └── big-lama.pt           # LaMa 修复模型（需手动下载）
├── input/                     # 输入文件目录
├── output/                    # 输出目录
├── output_webui/              # WebUI 输出目录
├── detection/                 # 检测中间结果
│   ├── frames/               # 原始帧
│   ├── masks/                # 检测掩码
│   └── previews/             # 预览图
└── 素材/                      # 素材资源
```

## 安装与设置

### 环境要求

- Python 3.10+
- pip（Python 包管理器）
- FFmpeg（视频处理必需，需添加到系统 PATH）
- NVIDIA GPU（推荐，用于 CUDA 加速）

### 安装步骤

1. 安装 Python 依赖：
```bash
pip install -r requirements.txt
```

2. 下载模型文件：
   - YOLO 模型：放置在 `models/yolo.pt`
   - LaMa 模型：从 [官方发布页](https://github.com/saic-mdal/lama/releases/download/v1.0/big-lama.pt) 下载并放置在 `models/big-lama.pt`

## 使用方法

### 1. WebUI 方式（推荐）

启动 WebUI：
```bash
python webui.py
```

然后在浏览器中访问 `http://127.0.0.1:7860`

### 2. 命令行方式

基本用法：
```bash
python main.py --input <输入文件或目录> --output <输出目录> [其他选项]
```

主要参数：
- `--model`：选择检测模型（yolo 或 florence）
- `--device`：运行设备（cuda 或 cpu）
- `--half-precision`：启用半精度推理
- `--video-batch-size`：视频处理批大小
- `--use-first-frame-detection`：视频使用首帧检测模式

YOLO 专用参数：
- `--yolo-model`：YOLO 模型路径
- `--conf-threshold`：置信度阈值
- `--iou-threshold`：IOU 阈值
- `--yolo-imgsz`：推理图像尺寸

Florence-2 专用参数：
- `--model-size`：模型大小（base 或 large）
- `--watermark-text`：水印文本提示

### 3. 单独使用 YOLO 检测

```bash
python step1_detect_watermark_yolo.py <输入文件> <输出目录> --model <YOLO模型路径>
```

## 性能优化建议

1. **使用 GPU**：确保 CUDA 环境配置正确
2. **首帧检测**：处理固定位置水印的视频时启用此选项
3. **调整批大小**：根据 GPU 显存适当调整 `video-batch-size`
4. **调整图像尺寸**：减小 YOLO 图像尺寸可提高检测速度

## 开发约定

- 使用 Python 3.10+ 语法
- 遵循 PEP 8 代码规范
- 使用 loguru 进行日志记录
- 使用 click 进行命令行参数解析
- 使用 PyTorch 进行深度学习模型推理

## 技术栈

- **深度学习框架**：PyTorch
- **计算机视觉**：OpenCV
- **Web 框架**：Gradio（WebUI）
- **模型库**：
  - ultralytics（YOLO）
  - transformers（Florence-2）
  - LaMa（图像修复）
- **工具库**：numpy, PIL, tqdm, loguru

## 常见问题

1. **模型文件缺失**：确保已下载并正确放置模型文件
2. **CUDA 不可用**：检查 NVIDIA 驱动和 CUDA Toolkit 安装
3. **FFmpeg 错误**：确保 FFmpeg 已安装并添加到系统 PATH
4. **显存不足**：减小批大小或禁用半精度推理