# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Apple Detection and Quality Grading Pipeline - a comprehensive computer vision system that detects apples in images/videos, tracks them across frames, and classifies their quality using deep learning models.

**Key Architecture Components:**
- **Detection**: YOLOv8-based apple detector (`src/models/apple_detector.py`)
- **Quality Classification**: MobileNetV3-based quality classifier (`src/models/quality_classifier.py`)
- **Tracking**: DeepSORT-based multi-object tracker (`src/models/apple_tracker.py`)
- **Pipeline**: Integrated processing pipeline (`src/pipeline/apple_pipeline.py`)
- **Web Interface**: Gradio-based web interface (`src/utils/gradio_interface.py`)

## Development Commands

### Setup and Validation
```bash
# Validate project setup (checks syntax, structure, configs)
python scripts/validate_setup.py

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all unit tests
python tests/run_tests.py

# Run specific test module
python -m pytest tests/unit/test_apple_detector.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Training Models
```bash
# Train detection model
python scripts/train_detection.py --config config/config.yaml --epochs 100 --batch-size 16

# Train quality classification model
python scripts/train_quality.py --config config/config.yaml --epochs 50 --batch-size 32

# Train complete pipeline
python scripts/train_pipeline.py --config config/config.yaml
```

### Running the Application
```bash
# Launch web interface
python src/utils/gradio_interface.py --share

# Process single image via pipeline
python -c "from src.pipeline.apple_pipeline import ApplePipeline; p = ApplePipeline(); result = p.process_image('path/to/image.jpg')"
```

## Code Architecture

### Multi-Stage Pipeline Design
The system follows a three-stage architecture:

1. **Detection Stage** (`AppleDetector`):
   - Uses YOLOv8 for apple detection in images
   - Configurable confidence/IoU thresholds
   - Returns bounding boxes with confidence scores

2. **Tracking Stage** (`AppleTracker`):
   - Implements DeepSORT for multi-object tracking across video frames
   - Maintains unique apple identities over time
   - Handles occlusions and re-identification

3. **Quality Assessment Stage** (`QualityClassificationPipeline`):
   - Uses MobileNetV3 for classifying apple quality
   - Three classes: good, minor_defect, major_defect
   - Processes cropped apple patches from detection boxes

### Key Integration Patterns
- **Configuration-Driven**: All models use `config/config.yaml` for hyperparameters
- **Device Agnostic**: Auto-detection of CUDA, MPS, or CPU with `device='auto'`
- **Result Objects**: Structured `PipelineResult` dataclass for consistent output
- **Modular Design**: Each component can be used independently or as integrated pipeline

### Data Flow
```
Image/Video → AppleDetector → AppleTracker (video only) → QualityClassifier → PipelineResult
```

## Configuration System

The project uses YAML-based configuration:

- **Main Config**: `config/config.yaml` - model params, training settings, performance targets
- **Dataset Config**: `config/dataset.yaml` - YOLO dataset configuration
- **Path Management**: All paths configured relative to project root

Key sections in `config.yaml`:
- `model.detection`: YOLOv8 detection parameters
- `model.quality_classification`: MobileNetV3 classification parameters
- `model.tracking`: DeepSORT tracking parameters
- `training`: Training hyperparameters for both models
- `performance`: Target metrics and benchmarks

## Testing Strategy

The project includes comprehensive unit tests for all major components:

- `tests/unit/test_apple_detector.py` - Detection model tests
- `tests/unit/test_quality_classifier.py` - Quality classification tests
- `tests/unit/test_apple_tracker.py` - Tracking functionality tests
- `tests/unit/test_pipeline.py` - End-to-end pipeline tests
- `tests/unit/test_data_processing.py` - Data processing utilities tests

Use `python tests/run_tests.py` for running all tests with detailed reporting.

## Model Training Workflow

1. **Data Preparation**: Use `DataManager` class to handle dataset splitting and YOLO format conversion
2. **Detection Training**: `train_detection.py` handles YOLOv8 training with automatic dataset YAML generation
3. **Quality Training**: `train_quality.py` handles MobileNetV3 training with classification dataset
4. **Validation**: Both scripts include automatic model evaluation and metrics reporting

## Performance Targets

The system is designed to meet specific performance benchmarks:
- Detection mAP50 > 90%
- Quality Classification Accuracy > 92%
- Real-time inference > 15 FPS
- Model size < 50MB for deployment

## Dependencies and Environment

Key dependencies include:
- **Core ML**: PyTorch 2.0+, torchvision, ultralytics (YOLOv8)
- **Computer Vision**: OpenCV, supervision, deep-sort-realtime
- **Web Interface**: Gradio, streamlit
- **Data Processing**: NumPy, pandas, albumentations, Pillow
- **Visualization**: matplotlib, seaborn, plotly

The project supports multiple compute backends (CUDA, MPS, CPU) with automatic detection.

## Development Guidelines

- All model classes inherit device management and configuration loading patterns
- Use the `ImageProcessor` class for consistent image preprocessing
- Follow the established result object patterns (`PipelineResult`, etc.)
- Maintain YAML configuration compatibility when adding new features
- Add corresponding unit tests for new components in `tests/unit/`