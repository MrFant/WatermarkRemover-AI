"""
第一步：使用 YOLO 模型检测水印位置并生成 mask
输入：图片或视频
输出：原始帧 + mask文件

优势：
- 速度快（比Florence快5-10倍）
- 可以自己训练模型
- 支持本地模型文件
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


def detect_watermark_with_yolo(image, model, conf_threshold=0.25, iou_threshold=0.45):
    """使用YOLO检测水印"""
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


def process_image(image_path, output_dir, model, conf_threshold, iou_threshold, max_bbox_percent, save_preview=True):
    """处理单张图片"""
    logger.info(f"处理图片: {image_path}")
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测水印
    detections = detect_watermark_with_yolo(image_rgb, model, conf_threshold, iou_threshold)
    
    # 创建mask
    mask = create_mask_from_detections(image_rgb.shape, detections, max_bbox_percent)
    
    # 保存
    frame_name = image_path.stem
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    frames_dir.mkdir(exist_ok=True, parents=True)
    masks_dir.mkdir(exist_ok=True, parents=True)
    
    if save_preview:
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(exist_ok=True, parents=True)
    
    cv2.imwrite(str(frames_dir / f"{frame_name}.png"), image)
    cv2.imwrite(str(masks_dir / f"{frame_name}.png"), mask)
    
    if save_preview:
        # 生成标注图（在原图上绘制检测框与遮罩叠加）
        annotated = image.copy()
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if np.any(mask > 127):
            mask_color = np.zeros_like(annotated)
            mask_color[mask > 127] = (255, 0, 0)  # 蓝色叠加区域
            annotated = cv2.addWeighted(annotated, 1.0, mask_color, 0.35, 0)
        
        cv2.imwrite(str(previews_dir / f"{frame_name}.png"), annotated)
    
    # 保存检测信息
    detection_info = {
        'frame': frame_name,
        'detections': detections,
        'mask_coverage': float(np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]))
    }
    
    return detection_info


def process_video(video_path, output_dir, model, conf_threshold, iou_threshold, max_bbox_percent, 
                 use_first_frame_detection=False, min_mask_coverage=0.0, save_preview=True):
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
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    frames_dir.mkdir(exist_ok=True, parents=True)
    masks_dir.mkdir(exist_ok=True, parents=True)
    
    if save_preview:
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存视频信息
    video_info = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'source_video': str(video_path),
        'detection_method': 'yolo'
    }
    
    with open(output_dir / "video_info.json", 'w') as f:
        json.dump(video_info, f, indent=2)
    
    # 如果使用第一帧检测
    first_frame_mask = None
    first_frame_detections = None
    
    if use_first_frame_detection:
        logger.info("🔍 使用第一帧检测模式（所有帧使用相同mask）")
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            first_frame_detections = detect_watermark_with_yolo(
                frame_rgb, model, conf_threshold, iou_threshold
            )
            first_frame_mask = create_mask_from_detections(
                frame_rgb.shape, first_frame_detections, max_bbox_percent
            )
            logger.info(f"第一帧检测到 {len(first_frame_detections)} 个目标")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        logger.info("🔍 每帧独立检测模式")
    
    # 处理所有帧
    frame_count = 0
    detection_infos = []
    
    desc = "使用统一检测" if use_first_frame_detection else "逐帧检测"
    
    with tqdm.tqdm(total=total_frames, desc=desc) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 获取mask
            if use_first_frame_detection and first_frame_mask is not None:
                mask = first_frame_mask
                detections = first_frame_detections
                filtered = False
            else:
                # 每帧独立检测
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect_watermark_with_yolo(
                    frame_rgb, model, conf_threshold, iou_threshold
                )
                mask = create_mask_from_detections(
                    frame_rgb.shape, detections, max_bbox_percent
                )
                
                # 计算覆盖率
                mask_coverage = float(np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]))
                
                # 过滤低覆盖率（可能是误检）
                if min_mask_coverage > 0 and mask_coverage * 100 < min_mask_coverage:
                    mask = np.zeros_like(mask)
                    mask_coverage = 0.0
                    filtered = True
                else:
                    filtered = False
            
            # 保存帧和mask
            frame_name = f"frame_{frame_count:06d}"
            cv2.imwrite(str(frames_dir / f"{frame_name}.png"), frame)
            cv2.imwrite(str(masks_dir / f"{frame_name}.png"), mask)
            
            if save_preview:
                # 生成标注图（在原图上绘制检测框与遮罩叠加）
                annotated = frame.copy()
                if detections:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                if np.any(mask > 127):
                    mask_color = np.zeros_like(annotated)
                    mask_color[mask > 127] = (255, 0, 0)
                    annotated = cv2.addWeighted(annotated, 1.0, mask_color, 0.35, 0)
                
                cv2.imwrite(str(previews_dir / f"{frame_name}.png"), annotated)
            
            # 记录检测信息
            detection_info = {
                'frame': frame_name,
                'frame_index': frame_count,
                'num_detections': len(detections) if detections else 0,
                'mask_coverage': float(np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])),
                'using_first_frame': use_first_frame_detection,
                'filtered': filtered if not use_first_frame_detection else False
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
    logger.info(f"📁 帧保存到: {frames_dir}")
    logger.info(f"📁 Mask保存到: {masks_dir}")
    if save_preview:
        logger.info(f"📁 预览保存到: {previews_dir}")
    
    # 输出统计
    if not use_first_frame_detection:
        logger.info("\n📊 检测统计:")
        
        coverages = [info['mask_coverage'] for info in detection_infos]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0
        
        logger.info(f"  平均mask覆盖: {avg_coverage*100:.2f}%")
        
        if min_mask_coverage > 0:
            filtered_count = sum(1 for info in detection_infos if info.get('filtered', False))
            if filtered_count > 0:
                logger.info(f"  🔍 过滤的帧: {filtered_count}")
        
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
@click.option("--use-first-frame/--detect-each-frame", default=False,
              help="视频是否使用第一帧检测（默认每帧检测）")
@click.option("--min-mask-coverage", type=float, default=0.0,
              help="最小mask覆盖率 (%%)，低于此值使用空mask")
def main(input_path, output_dir, model, conf_threshold, iou_threshold, 
         max_bbox_percent, device, use_first_frame, min_mask_coverage):
    """
    第一步：使用YOLO检测水印位置并生成mask
    
    输入：图片或视频文件
    输出：frames/ 目录（原始帧）+ masks/ 目录（mask）
    
    示例：
        python step1_detect_watermark_yolo.py input/video.mp4 output/ --model yolov8n.pt
        python step1_detect_watermark_yolo.py input/video.mp4 output/ --model custom_model.pt --conf-threshold 0.5
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
    
    # 加载YOLO模型
    yolo_model = load_yolo_model(model, device)
    
    # 判断输入类型
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
        # 视频
        process_video(input_path, output_dir, yolo_model, conf_threshold, iou_threshold,
                     max_bbox_percent, use_first_frame, min_mask_coverage)
    else:
        # 图片
        detection_info = process_image(input_path, output_dir, yolo_model,
                                      conf_threshold, iou_threshold, max_bbox_percent)
        
        if detection_info:
            with open(output_dir / "detection_info.json", 'w') as f:
                json.dump([detection_info], f, indent=2)
    
    logger.info(f"✅ 完成！输出目录: {output_dir}")
    logger.info(f"   - frames/: 原始帧")
    logger.info(f"   - masks/: 检测到的mask")
    logger.info(f"   - detection_info.json: 检测详细信息")


if __name__ == "__main__":
    main()

