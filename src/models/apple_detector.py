"""
Apple Detection Module using YOLOv8

This module implements the AppleDetector class for detecting apples in images
using YOLOv8 object detection model.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppleDetector:
    """
    YOLOv8-based Apple Detection Model
    
    This class provides functionality for detecting apples in images using
    YOLOv8 object detection model with optimized parameters for apple detection.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_size: str = 'n',
        device: str = 'auto',
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        use_minneapple_weights: bool = True
    ):
        """
        Initialize the Apple Detector

        Args:
            model_path: Path to pre-trained model weights. If None, uses YOLOv8 pretrained or MinneApple weights
            model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            use_minneapple_weights: Whether to use MinneApple pre-trained weights
        """
        self.device = self._setup_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_size = model_size
        self.use_minneapple_weights = use_minneapple_weights

        # Load model
        model_loaded = False

        # Try to load custom model path first
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading custom model from {model_path}")
            self.model = YOLO(model_path)
            self.model_source = f"Custom: {model_path}"
            model_loaded = True

        # Try MinneApple weights if enabled
        elif use_minneapple_weights:
            minneapple_paths = [
                'models/minneapple/Original_augmented_best.pt',
                'models/minneapple/Original_original_best.pt',
                'models/detection/minneapple_best.pt',
                'minneapple_best.pt'
            ]

            for path in minneapple_paths:
                if os.path.exists(path):
                    logger.info(f"Loading MinneApple pre-trained model from {path}")
                    self.model = YOLO(path)
                    self.model_source = f"MinneApple: {path}"
                    model_loaded = True
                    break

            if not model_loaded:
                logger.warning("MinneApple weights not found. Please download them from: https://github.com/joy0010/Apple-Detection-in-MinneApple-Dataset-with-YOLOv8")

        # Fallback to standard YOLOv8 pretrained
        if not model_loaded:
            logger.info(f"Loading YOLOv8{model_size} pretrained model")
            self.model = YOLO(f'yolov8{model_size}.pt')
            self.model_source = f"YOLOv8{model_size} COCO"

        # Move model to device
        self.model.to(self.device)

        # Performance metrics
        self.inference_times = []

        logger.info(f"Model loaded: {self.model_source}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Confidence threshold: {conf_threshold}")
        logger.info(f"IoU threshold: {iou_threshold}")
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        logger.info(f"Using device: {device}")
        return device
    
    def detect(
        self, 
        source: Union[str, np.ndarray, List[str]],
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Detect apples in image(s)
        
        Args:
            source: Input image(s) - can be path, numpy array, or list of paths
            save_results: Whether to save annotated images
            output_dir: Directory to save results
            
        Returns:
            List of detection results with bounding boxes, confidence scores, etc.
        """
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save_results,
            project=output_dir
        )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Process results
        processed_results = self._process_results(results)
        
        logger.info(f"Detection completed in {inference_time:.3f}s")
        logger.info(f"Detected {sum(len(r['boxes']) for r in processed_results)} apples")
        
        return processed_results
    
    def _process_results(self, results) -> List[Dict]:
        """
        Process raw YOLO results into structured format
        
        Args:
            results: Raw results from YOLO model
            
        Returns:
            List of processed detection results
        """
        processed_results = []
        
        for result in results:
            # Extract detection data
            boxes = result.boxes
            processed_result = {
                'image_path': result.path if hasattr(result, 'path') else None,
                'image_shape': result.orig_shape,
                'boxes': [],
                'confidence_scores': [],
                'class_ids': [],
                'total_apples': 0
            }
            
            if boxes is not None and len(boxes) > 0:
                # Convert tensors to numpy arrays
                xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
                conf = boxes.conf.cpu().numpy()  # Confidence scores
                cls = boxes.cls.cpu().numpy()   # Class IDs
                
                for i in range(len(xyxy)):
                    box_data = {
                        'x1': float(xyxy[i][0]),
                        'y1': float(xyxy[i][1]),
                        'x2': float(xyxy[i][2]),
                        'y2': float(xyxy[i][3]),
                        'center_x': float((xyxy[i][0] + xyxy[i][2]) / 2),
                        'center_y': float((xyxy[i][1] + xyxy[i][3]) / 2),
                        'width': float(xyxy[i][2] - xyxy[i][0]),
                        'height': float(xyxy[i][3] - xyxy[i][1]),
                        'area': float((xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]))
                    }
                    
                    processed_result['boxes'].append(box_data)
                    processed_result['confidence_scores'].append(float(conf[i]))
                    processed_result['class_ids'].append(int(cls[i]))
                
                processed_result['total_apples'] = len(xyxy)
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def detect_and_count(
        self, 
        source: Union[str, np.ndarray],
        return_annotated: bool = False
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Detect apples and return count with optional annotated image
        
        Args:
            source: Input image
            return_annotated: Whether to return annotated image
            
        Returns:
            Tuple of (apple_count, annotated_image)
        """
        results = self.detect(source)
        apple_count = results[0]['total_apples'] if results else 0
        
        annotated_image = None
        if return_annotated and results:
            annotated_image = self.annotate_image(source, results[0])
        
        return apple_count, annotated_image
    
    def annotate_image(
        self, 
        image: Union[str, np.ndarray], 
        detection_result: Dict,
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Annotate image with detection results
        
        Args:
            image: Input image (path or numpy array)
            detection_result: Detection result from detect() method
            show_confidence: Whether to show confidence scores
            color: Bounding box color (BGR)
            thickness: Line thickness
            
        Returns:
            Annotated image as numpy array
        """
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        # Draw bounding boxes
        for i, box in enumerate(detection_result['boxes']):
            x1, y1 = int(box['x1']), int(box['y1'])
            x2, y2 = int(box['x2']), int(box['y2'])
            confidence = detection_result['confidence_scores'][i]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            if show_confidence:
                label = f"Apple: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    img, 
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color, 
                    -1
                )
                cv2.putText(
                    img, 
                    label, 
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
        
        # Add total count
        total_text = f"Total Apples: {detection_result['total_apples']}"
        cv2.putText(
            img, 
            total_text, 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        return img
    
    def batch_detect(
        self, 
        image_paths: List[str],
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform batch detection on multiple images
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            
        Returns:
            List of detection results for each image
        """
        logger.info(f"Starting batch detection on {len(image_paths)} images")
        
        all_results = []
        for image_path in image_paths:
            try:
                results = self.detect(image_path, output_dir=output_dir)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        logger.info(f"Batch detection completed. Processed {len(all_results)} images")
        return all_results
    
    def train(
        self, 
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        save_dir: str = 'runs/detect',
        **kwargs
    ) -> object:
        """
        Train the detection model
        
        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_dir: Directory to save training results
            **kwargs: Additional training parameters
            
        Returns:
            Training results object
        """
        logger.info(f"Starting training with {epochs} epochs")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            device=self.device,
            project=save_dir,
            name='apple_detection',
            save_period=kwargs.get('save_period', 10),
            patience=kwargs.get('patience', 20),
            **kwargs
        )
        
        logger.info("Training completed")
        return results
    
    def validate(self, data_yaml: str = None) -> Dict:
        """
        Validate the model performance
        
        Args:
            data_yaml: Path to validation dataset YAML
            
        Returns:
            Validation metrics dictionary
        """
        logger.info("Starting model validation")
        
        if data_yaml:
            results = self.model.val(data=data_yaml)
        else:
            results = self.model.val()
        
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'F1': float(results.box.f1)
        }
        
        logger.info(f"Validation completed. mAP50: {metrics['mAP50']:.3f}")
        return metrics
    
    def export_model(
        self, 
        format: str = 'onnx',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export model to different formats for deployment
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            output_path: Output file path
            
        Returns:
            Path to exported model
        """
        logger.info(f"Exporting model to {format} format")
        
        exported_path = self.model.export(format=format, dynamic=True)
        
        if output_path:
            os.rename(exported_path, output_path)
            exported_path = output_path
        
        logger.info(f"Model exported to {exported_path}")
        return exported_path
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.inference_times = []
    
    def save_model(self, path: str):
        """Save model weights"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model = YOLO(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Example usage
    detector = AppleDetector(model_size='n', conf_threshold=0.3)
    
    # Example detection
    # results = detector.detect('path/to/image.jpg')
    # print(f"Detected {results[0]['total_apples']} apples")