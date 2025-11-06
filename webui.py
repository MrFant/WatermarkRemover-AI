import gradio as gr
from pathlib import Path
import subprocess
import os

def process_media(model, input_file, conf_threshold, iou_threshold, single_detection, yolo_imgsz, model_size, watermark_text, use_first_frame_mask, video_batch_size, resize_limit, expand, max_bbox_percent, half_precision):
    if input_file is None:
        return None, "Please upload a file."

    input_path = Path(input_file)
    output_dir = Path("output_webui")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    cmd = [
        "python", "main.py",
        "--input", str(input_path),
        "--output", str(output_dir),
        "--model", model,
        "--resize-limit", str(resize_limit),
        "--expand", str(expand),
        "--max-bbox-percent", str(max_bbox_percent),
    ]

    if half_precision:
        cmd.append("--half-precision")
    else:
        cmd.append("--no-half-precision")

    if model == 'yolo':
        cmd.extend([
            "--conf-threshold", str(conf_threshold),
            "--iou-threshold", str(iou_threshold),
            "--yolo-imgsz", str(yolo_imgsz),
        ])
        if single_detection:
            cmd.append("--single-detection")
    else: # florence
        cmd.extend([
            "--model-size", model_size,
            "--watermark-text", f'"{watermark_text}"'
        ])

    if input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        cmd.extend(["--video-batch-size", str(video_batch_size)])
        if use_first_frame_mask:
            cmd.append("--use-first-frame-detection")

    print(f"Running command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        status = f"Processing complete.\nOutput saved to {output_path}\n\n--- Log ---\n{process.stdout}\n{process.stderr}"
        return str(output_path), status
    except subprocess.CalledProcessError as e:
        status = f"An error occurred.\n\n--- Error Log ---\n{e.stdout}\n{e.stderr}"
        return None, status

with gr.Blocks() as demo:
    gr.Markdown("# Watermark Remover AI")
    gr.Markdown("Upload an image or video to remove the watermark.")

    model_selector = gr.Radio(["yolo", "florence"], value="yolo", label="Detection Model", info="选择用于水印检测的模型。YOLO通常更快，Florence-2更灵活。")

    with gr.Tabs():
        with gr.TabItem("Image"):
            with gr.Row():
                image_input = gr.Image(type="filepath", label="Input Image")
                image_output = gr.Image(label="Output Image")
            image_status = gr.Textbox(label="Status", lines=10)
            process_image_button = gr.Button("Remove Watermark from Image")

        with gr.TabItem("Video"):
            with gr.Row():
                video_input = gr.Video(label="Input Video")
                video_output = gr.Video(label="Output Video")
            video_status = gr.Textbox(label="Status", lines=10)
            process_video_button = gr.Button("Remove Watermark from Video")

    with gr.Accordion("General Processing Options", open=True):
        resize_limit = gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="Resize Limit", info="图像在送入修复模型前的最大尺寸。较小的值可以节省显存，但可能降低修复质量。")
        expand = gr.Slider(minimum=0, maximum=50, value=5, step=1, label="Expand Pixels", info="水印检测框向外扩展的像素数量。适当增加可确保水印被完全覆盖，但过大会影响周围内容。")
        max_bbox_percent = gr.Slider(minimum=1.0, maximum=50.0, value=10.0, label="Max Bbox Percent", info="单个水印检测框占图像总面积的最大百分比。超过此比例的框将被忽略，以避免误删非水印区域。")
        half_precision = gr.Checkbox(label="Use Half Precision (FP16)", value=True, info="启用半精度浮点运算（FP16），可减少显存使用并加速推理，但可能对精度有轻微影响。")

    with gr.Accordion("Model Specific Options", open=True):
        with gr.Column(visible=True) as yolo_options:
            conf_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, label="YOLO Confidence Threshold", info="YOLO模型检测水印的置信度阈值。值越高，检测结果越可靠，但可能漏掉不明显的水印。")
            iou_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.45, label="YOLO IOU Threshold", info="YOLO模型在非极大值抑制（NMS）中用于合并重叠检测框的交并比阈值。值越高，合并越严格。")
            single_detection = gr.Checkbox(label="YOLO Single Detection", value=False, info="YOLO模式下，是否只保留置信度最高的单个检测框。适用于图片中只有一个主要水印的场景。")
            yolo_imgsz = gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="YOLO Image Size (imgsz)", info="YOLO模型推理时的图像尺寸。较小的值可以加速检测，但可能影响小目标检测精度。")
        
        with gr.Column(visible=False) as florence_options:
            model_size = gr.Radio(["base", "large"], value="base", label="Florence-2 Model Size", info="Florence-2模型的大小。'large'模型效果更好但需要更多显存，'base'模型显存占用较少。")
            watermark_text = gr.Textbox(label="Florence Watermark Text Prompt", value="white English text watermark", info="Florence-2模型用于识别水印的文本提示。例如：'白色英文字符水印'。")

    with gr.Accordion("Video Specific Options", open=False):
        use_first_frame_mask = gr.Checkbox(label="Use First Frame Mask for Videos", value=False, info="视频处理时，是否只在第一帧检测水印并将其掩码应用于所有后续帧。可提高视频处理速度和一致性，但可能不适用于水印位置变化的视频。")
        video_batch_size = gr.Slider(minimum=1, maximum=64, value=8, step=1, label="Video Batch Size", info="处理视频时一次性读入内存并处理的帧数。更大的值可以提高GPU利用率，但需要更多显存。")

    def toggle_model_options(model):
        if model == "yolo":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    model_selector.change(toggle_model_options, model_selector, [yolo_options, florence_options])

    process_image_button.click(
        fn=process_media,
        inputs=[
            model_selector, image_input, conf_threshold, iou_threshold, single_detection, yolo_imgsz, 
            model_size, watermark_text, use_first_frame_mask, video_batch_size, resize_limit, expand, max_bbox_percent, half_precision
        ],
        outputs=[image_output, image_status]
    )

    process_video_button.click(
        fn=process_media,
        inputs=[
            model_selector, video_input, conf_threshold, iou_threshold, single_detection, yolo_imgsz, 
            model_size, watermark_text, use_first_frame_mask, video_batch_size, resize_limit, expand, max_bbox_percent, half_precision
        ],
        outputs=[video_output, video_status]
    )

if __name__ == "__main__":
    demo.launch()
