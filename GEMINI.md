# GEMINI.md

## Project Overview

This project is a powerful AI-powered tool for removing watermarks from images and videos. It leverages deep learning models to intelligently detect and inpaint watermarks.

The project is written in Python and utilizes the following core technologies:

*   **Detection:**
    *   **YOLO (You Only Look Once):** A fast and efficient object detection model, suitable for well-defined, static watermarks.
    *   **Florence-2:** A more flexible vision foundation model from Microsoft that can detect watermarks based on textual prompts (e.g., "a white logo").
*   **Inpainting:**
    *   **LaMa (Large Mask Inpainting):** A state-of-the-art image inpainting model that fills the area where the watermark was removed.
*   **User Interface:**
    *   **Gradio:** A Python library used to create a simple and intuitive web-based graphical user interface (WebUI).
    *   **Click:** A Python package for creating beautiful command-line interfaces.
*   **Core Libraries:**
    *   **PyTorch:** The primary deep learning framework.
    *   **OpenCV:** Used for image and video processing.
    *   **Ultralytics:** Provides the YOLO model implementation.
    *   **Transformers:** Provides the Florence-2 model implementation.

The architecture is split into two main entry points:
*   `main.py`: A command-line interface (CLI) that exposes all the tool's functionality and is ideal for batch processing and automation.
*   `webui.py`: A Gradio-based web application that provides a user-friendly interface for the tool, calling `main.py` as a subprocess.

## Building and Running

### 1. Prerequisites

*   Python 3.10+
*   `pip`
*   FFmpeg (must be in the system's PATH)
*   NVIDIA GPU with CUDA (recommended for performance)

### 2. Installation

1.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Models:**
    *   Download the **LaMa model** (`big-lama.pt`) and place it in the `models/` directory.
    *   Place your **YOLO model** (e.g., `yolo.pt`) in the `models/` directory.
    *   The Florence-2 model will be downloaded automatically on first use.

### 3. Running the Application

You can run the application in two ways:

*   **WebUI (Recommended for ease of use):**
    ```bash
    python webui.py
    ```
    This will start a local web server (usually at `http://127.0.0.1:7860`).

*   **Command-Line Interface (for automation and advanced use):**
    ```bash
    python main.py --input <path_to_file_or_dir> --output <output_dir> [OPTIONS]
    ```
    **Example:**
    ```bash
    python main.py --input ./input/my_video.mp4 --output ./output --model yolo --conf-threshold 0.5
    ```

## Development Conventions

*   **Modularity:** The core logic is contained within `main.py`, which is then used by `webui.py`. This separates the core functionality from the user interface.
*   **Configuration:** The application uses `argparse` in `main.py` for command-line argument parsing, which provides a clear and configurable interface. The WebUI maps its controls directly to these command-line arguments.
*   **Dependencies:** Project dependencies are clearly listed in `requirements.txt`.
*   **Logging:** The `loguru` library is used for clear and informative logging, including performance statistics.
*   **Batch Processing:** The tool is designed for efficiency, with batch processing implemented for video frames to leverage GPU parallelism.
*   **Hardware Acceleration:** The tool supports CUDA for GPU acceleration and can use NVENC for hardware-accelerated video encoding via FFmpeg.
