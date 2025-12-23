"""
YOLO调试器 - 在图片中标注YOLO识别的所有物体的方框和置信度等信息
输入：图片或视频
输出：标注了YOLO识别结果的图片

优势：
- 速度快（比Florence快5-10倍）
- 可以自己训练模型
- 支持本地模型文件
- 支持检测所有物体类别
"""

import sys
import click
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger
import tqdm
import json
import time

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("未安装 ultralytics 库，请运行: pip install ultralytics")
    sys.exit(1)


def load_yolo_model(model_path, device='cuda'):
    """加载YOLO模型"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        sys.exit(1)
    
    logger.info(f"加载YOLO模型: {model_path}")
    start_time = time.time()
    
    try:
        model = YOLO(str(model_path))
        
        # 设置设备
        if device == 'cuda':
            model.to('cuda')
        else:
            model.to('cpu')
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成 ({load_time:.2f}秒)")
        
        # 显示模型信息
        if hasattr(model, 'names'):
            logger.info(f"检测类别: {model.names}")
        
        return model
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        sys.exit(1)


def detect_all_objects(image, model, conf_threshold=0.25, iou_threshold=0.45):
    """使用YOLO检测所有物体"""
    # YOLO推理
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    detections = []
    
    if results and len(results) > 0:
        result = results[0]
        
        # 获取检测框
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()   # 置信度
            classes = result.boxes.cls.cpu().numpy()  # 类别
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': model.names[int(cls)] if hasattr(model, 'names') else str(int(cls))
                })
    
    return detections


def create_mask_from_detections(image_shape, detections, max_bbox_percent=10.0, expand_pixels=5):
    """从YOLO检测结果创建mask"""
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not detections:
        return mask
    
    image_area = width * height
    
    logger.debug(f"创建mask，检测到 {len(detections)} 个目标")
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        bbox_area = (x2 - x1) * (y2 - y1)
        area_percent = (bbox_area / image_area) * 100
        
        logger.debug(f"  [{i+1}] {det['class_name']} (conf: {det['confidence']:.2f})")
        logger.debug(f"      位置: ({x1}, {y1}) -> ({x2}, {y2})")
        logger.debug(f"      占比: {area_percent:.2f}%")
        
        # 检查大小限制
        if area_percent <= max_bbox_percent:
            # 扩展边界
            x1 = max(0, x1 - expand_pixels)
            y1 = max(0, y1 - expand_pixels)
            x2 = min(width, x2 + expand_pixels)
            y2 = min(height, y2 + expand_pixels)
            
            mask[y1:y2, x1:x2] = 255
            logger.debug(f"      ✅ 已添加到mask")
        else:
            logger.warning(f"      ❌ 跳过（超过最大占比 {max_bbox_percent}%）")
    
    # 后处理：膨胀和模糊
    if np.any(mask > 0):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask


def annotate_image(image, detections, show_confidence=True, show_class_name=True):
    """在图片上标注YOLO检测结果"""
    annotated = image.copy()
    
    if detections:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # 绘制矩形框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 准备标注文本
            label_parts = []
            if show_class_name:
                label_parts.append(det['class_name'])
            if show_confidence:
                label_parts.append(f"{det['confidence']:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                # 绘制文本背景
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - text_height - 5)), (x1 + text_width, y1), (0, 0, 255), -1)
                # 绘制文本
                cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    return annotated


def process_image(image_path, output_dir, model, conf_threshold, iou_threshold, max_bbox_percent, show_confidence=True, show_class_name=True):
    """处理单张图片"""
    logger.info(f"处理图片: {image_path}")
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测所有物体
    detections = detect_all_objects(image_rgb, model, conf_threshold, iou_threshold)
    
    # 标注图片
    annotated_image = annotate_image(image, detections, show_confidence, show_class_name)
    
    # 保存
    frame_name = image_path.stem
    output_image_path = output_dir / f"{frame_name}_annotated.png"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cv2.imwrite(str(output_image_path), annotated_image)
    logger.info(f"标注图片已保存到: {output_image_path}")
    
    # 创建mask（可选）
    mask = create_mask_from_detections(image_rgb.shape, detections, max_bbox_percent)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(masks_dir / f"{frame_name}.png"), mask)
    
    # 保存检测信息
    detection_info = {
        'frame': frame_name,
        'detections': detections,
        'mask_coverage': float(np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]))
    }
    
    return detection_info


def process_video(video_path, output_dir, model, conf_threshold, iou_threshold, max_bbox_percent, 
                 show_confidence=True, show_class_name=True):
    """处理视频"""
    logger.info(f"处理视频: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 创建输出目录
    output_frames_dir = output_dir / "frames_annotated"
    output_frames_dir.mkdir(exist_ok=True, parents=True)
    
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True, parents=True)
    
    # 处理所有帧
    frame_count = 0
    detection_infos = []
    
    with tqdm.tqdm(total=total_frames, desc="逐帧检测和标注") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测水印
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# 检测所有物体
            detections = detect_all_objects(
                frame_rgb, model, conf_threshold, iou_threshold
            )
            
            # 标注图片
            annotated_frame = annotate_image(frame, detections, show_confidence, show_class_name)
            
            # 保存标注后的帧
            frame_name = f"frame_{frame_count:06d}"
            cv2.imwrite(str(output_frames_dir / f"{frame_name}.png"), annotated_frame)
            
            # 创建mask（可选）
            mask = create_mask_from_detections(
                frame_rgb.shape, detections, max_bbox_percent
            )
            cv2.imwrite(str(masks_dir / f"{frame_name}.png"), mask)
            
            # 记录检测信息
            detection_info = {
                'frame': frame_name,
                'frame_index': frame_count,
                'num_detections': len(detections) if detections else 0,
                'mask_coverage': float(np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]))
            }
            
            detection_infos.append(detection_info)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    # 保存检测信息
    with open(output_dir / "detection_info.json", 'w') as f:
        json.dump(detection_infos, f, indent=2)
    
    # 统计信息
    logger.info(f"\n✅ 完成！共处理 {frame_count} 帧")
    logger.info(f"📁 标注帧保存到: {output_frames_dir}")
    logger.info(f"📁 Mask保存到: {masks_dir}")
    
    # 输出统计
    coverages = [info['mask_coverage'] for info in detection_infos]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    
    logger.info(f"📊 检测统计:")
    logger.info(f"  平均mask覆盖: {avg_coverage*100:.2f}%")
    
    num_detections = [info['num_detections'] for info in detection_infos]
    unique_counts = set(num_detections)
    
    if len(unique_counts) > 1:
        logger.warning(f"  ⚠️  检测数量不一致:")
        for count in sorted(unique_counts):
            frames_with_count = sum(1 for n in num_detections if n == count)
            logger.warning(f"     {count}个目标: {frames_with_count}帧")
    else:
        logger.info(f"  ✅ 所有帧检测一致: {list(unique_counts)[0]}个目标")


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--model", type=click.Path(exists=True), required=True,
              help="YOLO模型文件路径 (.pt)")
@click.option("--conf-threshold", type=float, default=0.25,
              help="置信度阈值 (0.0-1.0)")
@click.option("--iou-threshold", type=float, default=0.45,
              help="IOU阈值 (0.0-1.0)")
@click.option("--max-bbox-percent", type=float, default=10.0,
              help="最大边界框占比 (%)")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default=None,
              help="运行设备")
@click.option("--no-confidence", is_flag=True,
              help="不显示置信度")
@click.option("--no-class-name", is_flag=True,
              help="不显示类别名称")
def main(input_path, output_dir, model, conf_threshold, iou_threshold, 
         max_bbox_percent, device, no_confidence, no_class_name):
    """
    YOLO调试器 - 在图片中标注YOLO识别的方框和置信度等信息
    
    输入：图片或视频文件
    输出：标注了YOLO识别结果的图片
    
    示例：
        python yolo_debugger.py input/video.mp4 output/ --model yolov8n.pt
        python yolo_debugger.py input/video.mp4 output/ --model custom_model.pt --conf-threshold 0.5
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 设置设备
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"设备: {device}")
    logger.info(f"置信度阈值: {conf_threshold}")
    logger.info(f"IOU阈值: {iou_threshold}")
    logger.info(f"显示置信度: {not no_confidence}")
    logger.info(f"显示类别名称: {not no_class_name}")
    
    # 加载YOLO模型
    yolo_model = load_yolo_model(model, device)
    
    # 判断输入类型
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
        # 视频
        process_video(input_path, output_dir, yolo_model, conf_threshold, iou_threshold,
                     max_bbox_percent, not no_confidence, not no_class_name)
    else:
        # 图片
        detection_info = process_image(input_path, output_dir, yolo_model,
                                      conf_threshold, iou_threshold, max_bbox_percent,
                                      not no_confidence, not no_class_name)
        
        if detection_info:
            with open(output_dir / "detection_info.json", 'w') as f:
                json.dump([detection_info], f, indent=2)
    
    logger.info(f"✅ 完成！输出目录: {output_dir}")
    logger.info(f"   - 标注图片已保存到输出目录")
    logger.info(f"   - masks/: 检测到的mask")
    logger.info(f"   - detection_info.json: 检测详细信息")


if __name__ == "__main__":
    main()
