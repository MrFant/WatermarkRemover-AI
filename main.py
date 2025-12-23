
import os
import sys
import json
import cv2
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Union
import click
import numpy as np
from PIL import Image
from loguru import logger

try:
    from tqdm import tqdm
except ImportError:
    logger.error("未安装 tqdm 库，请运行: pip install tqdm")
    tqdm = None

import torch
from torch.cuda.amp import autocast
import argparse
import torchvision.transforms.functional as TF

# Local imports


# --- Model Imports ---
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("未安装 ultralytics 库，请运行: pip install ultralytics")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from enum import Enum
except ImportError:
    logger.error("未安装 transformers 库，请运行: pip install transformers")
    sys.exit(1)


# --- Global Settings ---
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
IMAGE_EXTS = { ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = { ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
LAMA_MODEL_PATH = Path("models") / "big-lama.pt"

# --- Timing Class ---
class Timing:
    def __init__(self):
        self.total_start = time.perf_counter()
        self.model_load_secs = 0.0
        self.lama_load_secs = 0.0
        self.io_read_secs = 0.0
        self.io_write_secs = 0.0
        self.detect_secs = 0.0
        self.mask_make_secs = 0.0
        self.lama_inpaint_secs = 0.0
        self.audio_merge_secs = 0.0
        self.images = 0
        self.frames = 0
    def summary(self):
        total_time = time.perf_counter() - self.total_start
        logger.info("\n⏱️ 性能统计汇总:")
        logger.info(f"  总耗时: {total_time:.2f}s")
        logger.info(f"  模型加载: 检测模型 {self.model_load_secs:.2f}s, LaMa {self.lama_load_secs:.2f}s")
        logger.info(f"  I/O 读取: {self.io_read_secs:.2f}s, I/O 写入: {self.io_write_secs:.2f}s")
        logger.info(f"  检测: {self.detect_secs:.2f}s, 掩码生成: {self.mask_make_secs:.2f}s")
        logger.info(f"  LaMa修复: {self.lama_inpaint_secs:.2f}s, 音频合并: {self.audio_merge_secs:.2f}s")
        if self.images > 0:
            avg_img = (self.detect_secs + self.mask_make_secs + self.lama_inpaint_secs) / self.images
            logger.info(f"  图片数: {self.images}, 平均每图: {avg_img:.2f}s")
        if self.frames > 0:
            avg_frame = (self.detect_secs + self.mask_make_secs + self.lama_inpaint_secs) / self.frames
            logger.info(f"  视频帧数: {self.frames}, 平均每帧: {avg_frame:.3f}s")

TIMING = None

# --- Florence-2 Specific Functions ---
class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"

def enhance_contrast(image: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    cl_sharpened = cv2.filter2D(cl, -1, kernel)
    enhanced_lab = cv2.merge((cl_sharpened, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

def identify(task_prompt: TaskType, image: np.ndarray, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.shape[1], image.shape[0])
    )

def get_mask_florence(image: np.ndarray, model, processor, device: str, args) -> np.ndarray:
    t0 = time.perf_counter()
    enh_image = enhance_contrast(image)
    with autocast(enabled=args.half_precision):
        parsed_answer = identify(TaskType.OPEN_VOCAB_DETECTION, enh_image, args.watermark_text, model, processor, device)
    TIMING.detect_secs += time.perf_counter() - t0
    
    t_mask = time.perf_counter()
    mask_np = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.shape[1] * image.shape[0]
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= args.max_bbox_percent:
                x1, y1, x2, y2 = max(0, x1 - args.expand), max(0, y1 - args.expand), min(image.shape[1], x2 + args.expand), min(image.shape[0], y2 + args.expand)
                mask_np[y1:y2, x1:x2] = 255
    
    kernel = np.ones((3, 3), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)
    TIMING.mask_make_secs += time.perf_counter() - t_mask
    return mask_np

# --- YOLO Specific Functions ---
def parse_yolo_results(result, names_map: dict) -> List[dict]:
    dets = []
    if getattr(result, "boxes", None) is not None:
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            dets.append({"bbox": [int(b) for b in box], "confidence": float(conf), "class_name": names_map.get(int(cls), "N/A")})
    return dets

def get_mask_yolo(image: np.ndarray, model: YOLO, args) -> np.ndarray:
    t0 = time.perf_counter()
    results = model.predict([image], imgsz=args.yolo_imgsz, conf=args.conf_threshold, iou=args.iou_threshold, max_det=(1 if args.single_detection else 300), verbose=False)
    TIMING.detect_secs += time.perf_counter() - t0
    
    t_mask = time.perf_counter()
    dets = parse_yolo_results(results[0], model.names)
    if args.single_detection and dets:
        dets = [max(dets, key=lambda d: d["confidence"])]
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    img_area = image.shape[0] * image.shape[1]
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        if img_area > 0 and ((x2 - x1) * (y2 - y1) / img_area) * 100.0 > args.max_bbox_percent:
            continue
        x1, y1, x2, y2 = max(0, x1 - args.expand), max(0, y1 - args.expand), min(image.shape[1], x2 + args.expand), min(image.shape[0], y2 + args.expand)
        mask[y1:y2, x1:x2] = 255
    TIMING.mask_make_secs += time.perf_counter() - t_mask
    return mask

# --- Core Processing Functions ---
def load_models(args):
    t0 = time.perf_counter()
    if args.model == 'yolo':
        if not Path(args.yolo_model).exists():
            logger.error(f"YOLO 模型文件不存在: {args.yolo_model}")
            sys.exit(1)
        logger.info(f"加载 YOLO 模型: {args.yolo_model}")
        detection_model = YOLO(str(args.yolo_model))
        detection_model.to(args.device)
        detection_processor = None
    else: # florence
        model_name = f"microsoft/Florence-2-{args.model_size}"
        logger.info(f"加载 Florence-2 {args.model_size} 模型...")
        detection_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, local_files_only=True).to(args.device).eval()
        detection_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if args.half_precision:
            detection_model = detection_model.half()
    TIMING.model_load_secs = time.perf_counter() - t0
    logger.info(f"检测模型加载完成，用时 {TIMING.model_load_secs:.2f}s")
    
    # Load LaMa model directly from Torch Script
    t_lama = time.perf_counter()
    logger.info(f"加载 LaMa Torch Script 模型: {LAMA_MODEL_PATH}")
    if not LAMA_MODEL_PATH.exists():
        logger.error(f"LaMa 模型文件不存在: {LAMA_MODEL_PATH}")
        logger.error("请将 big-lama.pt 文件放置在 models/ 目录下。")
        sys.exit(1)
    
    lama_model = torch.load(LAMA_MODEL_PATH, map_location=args.device, weights_only=False)
    lama_model.to(args.device)
    lama_model.eval()
    TIMING.lama_load_secs = time.perf_counter() - t_lama
    logger.info(f"LaMa 加载完成，用时 {TIMING.lama_load_secs:.2f}s")
    
    return detection_model, detection_processor, lama_model

def inpaint_with_lama_batch(images_rgb: List[np.ndarray], masks: List[np.ndarray], lama_model, args) -> List[np.ndarray]:
    t0 = time.perf_counter()
    
    # 1. Pre-processing
    batch_size = len(images_rgb)
    original_h, original_w = images_rgb[0].shape[:2]
    
    # Adjust size to be multiples of 8 (required by LaMa model)
    h = ((original_h + 7) // 8) * 8
    w = ((original_w + 7) // 8) * 8
    
    batch_images = torch.zeros(batch_size, 3, h, w, dtype=torch.float32)
    batch_masks = torch.zeros(batch_size, 1, h, w, dtype=torch.float32)

    for i in range(batch_size):
        # Resize image to multiple of 8
        img = Image.fromarray(images_rgb[i])
        img_resized = img.resize((w, h), Image.BICUBIC)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        # 确保图像值在合理范围内
        img_np = np.clip(img_np, 0.0, 1.0)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        
        # Resize mask to multiple of 8
        mask = Image.fromarray(masks[i])
        mask_resized = mask.resize((w, h), Image.BICUBIC)
        mask_np = np.array(mask_resized).astype(np.float32) / 255.0
        # 二值化处理，确保掩码只有0和1
        mask_np = (mask_np > 0.5).astype(np.float32)
        # 检查掩码覆盖率，避免完全覆盖图像
        mask_coverage = np.mean(mask_np)
        if mask_coverage > 0.95:
            logger.warning(f"掩码覆盖率过高 ({mask_coverage:.2f})，可能导致修复效果不佳")
            # 降低掩码覆盖率
            mask_np = np.minimum(mask_np, 0.95)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        batch_images[i] = img_tensor
        batch_masks[i] = mask_tensor

    masked_images = batch_images * (1 - batch_masks)
    # 确保 masked_images 不是全黑
    for i in range(batch_size):
        img_min = masked_images[i].min().item()
        img_max = masked_images[i].max().item()
        if img_max - img_min < 0.01:
            logger.warning(f"第 {i} 个图像的 masked_images 接近全黑，可能导致修复失败")
            # 添加少量噪声，避免模型输出全黑
            masked_images[i] += 0.01 * torch.randn_like(masked_images[i])
            masked_images[i] = torch.clamp(masked_images[i], 0.0, 1.0)
    
    masked_images = masked_images.to(args.device)
    batch_masks = batch_masks.to(args.device)

    # 2. Inference (修复了黑块 Bug)
    # 必须强制使用 FP32 (autocast enabled=False)，否则 FFC 层会计算出 NaN
    with torch.no_grad(), autocast(enabled=False):
        inpainted_batch = lama_model(masked_images, batch_masks)
        # 检查模型输出是否包含 NaN 或 Inf
        if torch.isnan(inpainted_batch).any() or torch.isinf(inpainted_batch).any():
            logger.error("LaMa 模型输出包含 NaN 或 Inf")
            # 使用原始图像作为回退
            inpainted_batch = masked_images.clone()

    # 3. Post-processing
    inpainted_results = []
    for i in range(batch_size):
        output_img = inpainted_batch[i].permute(1, 2, 0).cpu().numpy()
        # 确保输出值在 [0, 1] 范围内，防止溢出
        output_img = np.clip(output_img, 0.0, 1.0)
        output_img = (output_img * 255).astype(np.uint8)
        
        # 检查输出是否全黑
        if np.mean(output_img) < 10:
            logger.warning(f"第 {i} 个图像修复结果接近全黑，使用原始图像回退")
            # 使用原始图像的缩放版本作为回退
            img_pil = Image.fromarray(images_rgb[i])
            output_img = np.array(img_pil.resize((original_w, original_h), Image.BICUBIC))
        
        # Resize back to original size
        output_img_pil = Image.fromarray(output_img)
        output_img_resized = output_img_pil.resize((original_w, original_h), Image.BICUBIC)
        inpainted_results.append(np.array(output_img_resized))

    TIMING.lama_inpaint_secs += time.perf_counter() - t0
    return inpainted_results


def process_image(path: Path, detection_model, detection_processor, lama_model, args):
    logger.info(f"处理图片: {path.name}")
    t_r0 = time.perf_counter()
    img_bgr = cv2.imread(str(path))
    TIMING.io_read_secs += time.perf_counter() - t_r0
    if img_bgr is None:
        logger.error(f"无法读取图片: {path}"); return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if args.model == 'yolo':
        mask = get_mask_yolo(img_rgb, detection_model, args)
    else:
        mask = get_mask_florence(img_rgb, detection_model, detection_processor, args.device, args)

    if not np.any(mask):
        result_rgb = img_rgb
    else:
        # Use the batch function with a batch size of 1
        result_rgb = inpaint_with_lama_batch([img_rgb], [mask], lama_model, args)[0]
    
    # Add suffix to output filename to avoid overwriting original
    out_filename = path.stem + "_no_watermark" + path.suffix
    out_path = Path(args.output) / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_w0 = time.perf_counter()
    cv2.imwrite(str(out_path), cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    TIMING.io_write_secs += time.perf_counter() - t_w0
    TIMING.images += 1

def process_video(path: Path, detection_model, detection_processor, lama_model, args):
    logger.info(f"处理视频: {path.name}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {path}"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = tmp_dir / "tmp_no_audio.mp4"
    out = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    pbar = tqdm(total=total_frames, desc=f"处理 {path.name}", unit="frame") if tqdm and total_frames > 0 else None
    
    batch_frames_rgb = []
    last_mask = None

    while True:
        t_r0 = time.perf_counter()
        ret, frame_bgr = cap.read()
        TIMING.io_read_secs += time.perf_counter() - t_r0
        if ret:
            batch_frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        is_batch_full = len(batch_frames_rgb) == args.video_batch_size
        is_last_batch = (not ret) and (len(batch_frames_rgb) > 0)

        if is_batch_full or is_last_batch:
            # --- Batch Detection ---
            batch_masks = []
            if args.use_first_frame_detection and last_mask is not None:
                batch_masks = [last_mask] * len(batch_frames_rgb)
            else:
                if args.model == 'yolo':
                    # Run YOLO detection on the entire batch at once
                    t0 = time.perf_counter()
                    yolo_results = detection_model.predict(batch_frames_rgb, imgsz=args.yolo_imgsz, conf=args.conf_threshold, iou=args.iou_threshold, verbose=False)
                    TIMING.detect_secs += time.perf_counter() - t0

                    # Create masks from the batch results
                    t_mask = time.perf_counter()
                    for i, result in enumerate(yolo_results):
                        dets = parse_yolo_results(result, detection_model.names)
                        if args.single_detection and dets:
                            dets = [max(dets, key=lambda d: d["confidence"])]
                        
                        mask = np.zeros(batch_frames_rgb[i].shape[:2], dtype=np.uint8)
                        img_area = batch_frames_rgb[i].shape[0] * batch_frames_rgb[i].shape[1]
                        for det in dets:
                            x1, y1, x2, y2 = det["bbox"]
                            if img_area > 0 and ((x2 - x1) * (y2 - y1) / img_area) * 100.0 > args.max_bbox_percent:
                                continue
                            x1, y1, x2, y2 = max(0, x1 - args.expand), max(0, y1 - args.expand), min(batch_frames_rgb[i].shape[1], x2 + args.expand), min(batch_frames_rgb[i].shape[0], y2 + args.expand)
                            mask[y1:y2, x1:x2] = 255
                        batch_masks.append(mask)
                    TIMING.mask_make_secs += time.perf_counter() - t_mask
                    
                    if batch_masks:
                        last_mask = batch_masks[-1]
                else: # Florence (remains single-frame processing in loop)
                    for frame_rgb in batch_frames_rgb:
                        mask = get_mask_florence(frame_rgb, detection_model, detection_processor, args.device, args)
                        batch_masks.append(mask)
                        last_mask = mask
            
            # --- Batch Inpainting ---
            frames_to_inpaint = []
            masks_to_inpaint = []
            inpaint_indices = []

            for i, (frame, mask) in enumerate(zip(batch_frames_rgb, batch_masks)):
                if np.any(mask):
                    frames_to_inpaint.append(frame)
                    masks_to_inpaint.append(mask)
                    inpaint_indices.append(i)
            
            final_batch_frames = list(batch_frames_rgb)
            if frames_to_inpaint:
                inpainted_frames = inpaint_with_lama_batch(frames_to_inpaint, masks_to_inpaint, lama_model, args)
                # Splice inpainted frames back into the batch
                for i, original_index in enumerate(inpaint_indices):
                    final_batch_frames[original_index] = inpainted_frames[i]

            # --- Batch Write ---
            t_w0 = time.perf_counter()
            for frame_rgb in final_batch_frames:
                out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            TIMING.io_write_secs += time.perf_counter() - t_w0

            if pbar: pbar.update(len(batch_frames_rgb))
            TIMING.frames += len(batch_frames_rgb)
            batch_frames_rgb.clear()

        if not ret: break

    if pbar: pbar.close()
    cap.release(); out.release()

    # Add suffix to output filename to avoid overwriting original
    out_filename = path.stem + "_no_watermark" + path.suffix
    output_path = Path(args.output) / out_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Check for hardware acceleration
        video_codec = "libx264"
        if args.device == "cuda":
            try:
                subprocess.check_output(["ffmpeg", "-h", "encoder=h264_nvenc"], stderr=subprocess.STDOUT)
                video_codec = "h264_nvenc"
                logger.info("使用 NVIDIA GPU (h264_nvenc) 进行硬件加速视频编码。")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("未找到 h264_nvenc 编码器，将使用CPU (libx264) 进行视频编码。")

        cmd = ["ffmpeg", "-y", "-i", str(tmp_video), "-i", str(path), "-map", "0:v:0", "-map", "1:a?", "-c:v", video_codec, "-c:a", "copy", str(output_path)]
        t_aud = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        TIMING.audio_merge_secs += time.perf_counter() - t_aud
        logger.info(f"已合并音频并输出: {output_path.name}")
    except Exception as e:
        logger.warning(f"合并音频失败: {e}. 将复制无音频视频。")
        shutil.copyfile(tmp_video, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def main():
    global TIMING
    TIMING = Timing()
    torch.set_grad_enabled(False)
    cv2.setNumThreads(0)

    parser = argparse.ArgumentParser(description="通用水印去除管线 (YOLO & Florence)")
    # --- General Args ---
    parser.add_argument("--input", type=str, default=str(INPUT_DIR), help="输入路径：目录或单个文件")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--model", type=str, choices=["yolo", "florence"], default="yolo", help="选择检测模型")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--half-precision", action="store_true", default=False, help="启用半精度(FP16)")
    parser.add_argument("--no-half-precision", dest="half_precision", action="store_false", help="禁用半精度(FP16)")
    parser.add_argument("--resize-limit", type=int, default=768, help="LaMa hd_strategy_resize_limit")
    parser.add_argument("--expand", type=int, default=5, help="掩码扩张像素")
    parser.add_argument("--max-bbox-percent", type=float, default=10.0, help="单框最大占比(%%) 超过即跳过")
    # --- Video Args ---
    parser.add_argument("--video-batch-size", type=int, default=8, help="处理视频时的批大小")
    parser.add_argument("--use-first-frame-detection", action="store_true", default=False, help="视频使用首帧检测掩码")
    # --- YOLO Args ---
    parser.add_argument("--yolo-model", type=str, default="models/yolo.pt", help="YOLO 模型文件路径")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="YOLO IOU阈值")
    parser.add_argument("--single-detection", default=True, action="store_true", help="仅保留最高置信度的一个检测框")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO模型推理时的图像尺寸。较小的值可以加速检测。")
    # --- Florence Args ---
    parser.add_argument("--model-size", type=str, choices=["base", "large"], default="base", help="Size of Florence-2 model (base uses less memory)")
    parser.add_argument("--watermark-text", type=str, default="white English text watermark", help="Text prompt for watermark detection (default: 'white English text watermark')")

    args = parser.parse_args()
    
    logger.info(f"设备: {args.device}, 检测模型: {args.model}")
    if args.device == "cuda": torch.backends.cudnn.benchmark = True

    model, processor, lama_manager = load_models(args)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}"); sys.exit(1)

    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    else:
        files_to_process.extend([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)])

    for path in files_to_process:
        try:
            if path.suffix.lower() in IMAGE_EXTS:
                process_image(path, model, processor, lama_manager, args)
            elif path.suffix.lower() in VIDEO_EXTS:
                process_video(path, model, processor, lama_manager, args)
        except Exception as e:
            logger.error(f"处理文件失败: {path.name}: {e}")

    logger.info(f"✅ 全部完成！输出目录: {args.output}")
    TIMING.summary()

if __name__ == "__main__":
    main()
