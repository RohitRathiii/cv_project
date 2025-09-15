"""
Apple Detection and Quality Grading Pipeline

This module implements the main ApplePipeline class that integrates
detection, tracking, and quality assessment into a complete system.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import cv2
import torch
import yaml
from dataclasses import dataclass
import json

# Import custom modules
from ..models.apple_detector import AppleDetector
from ..models.quality_classifier import QualityClassificationPipeline
from ..models.apple_tracker import AppleTracker
from ..data.data_processing import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Data structure for pipeline results"""
    total_apples: int
    active_tracks: int
    unique_apples: int
    quality_distribution: Dict[str, int]
    quality_score: float
    confidence_scores: List[float]
    processing_time: float
    annotated_image: Optional[np.ndarray] = None


class ApplePipeline:
    """
    Complete Apple Detection and Quality Grading Pipeline
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        detection_model_path: Optional[str] = None,
        quality_model_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """Initialize the Apple Pipeline"""
        # Load configuration
        self.config = self._load_config(config_path)
        self.device = self._setup_device(device)
        
        # Initialize components
        self._initialize_detector(detection_model_path)
        self._initialize_quality_classifier(quality_model_path)
        self._initialize_tracker()
        self._initialize_image_processor()
        
        # Pipeline state
        self.is_video_mode = False
        self.frame_count = 0
        self.processing_times = []
        
        logger.info("Apple Pipeline initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'model': {
                    'detection': {'confidence_threshold': 0.3, 'iou_threshold': 0.5},
                    'quality_classification': {'input_size': [224, 224]},
                    'tracking': {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3}
                }
            }
        return config
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        logger.info(f"Using device: {device}")
        return device
    
    def _initialize_detector(self, model_path: Optional[str]):
        """Initialize apple detector"""
        detector_config = self.config.get('model', {}).get('detection', {})
        
        self.detector = AppleDetector(
            model_path=model_path,
            device=self.device,
            conf_threshold=detector_config.get('confidence_threshold', 0.3),
            iou_threshold=detector_config.get('iou_threshold', 0.5)
        )
    
    def _initialize_quality_classifier(self, model_path: Optional[str]):
        """Initialize quality classifier"""
        classifier_config = self.config.get('model', {}).get('quality_classification', {})
        
        self.quality_classifier = QualityClassificationPipeline(
            device=self.device,
            input_size=tuple(classifier_config.get('input_size', [224, 224]))
        )
        
        if model_path and os.path.exists(model_path):
            self.quality_classifier.load_model(model_path)
    
    def _initialize_tracker(self):
        """Initialize apple tracker"""
        tracker_config = self.config.get('model', {}).get('tracking', {})
        
        self.tracker = AppleTracker(
            max_age=tracker_config.get('max_age', 30),
            min_hits=tracker_config.get('min_hits', 3),
            iou_threshold=tracker_config.get('iou_threshold', 0.3)
        )
    
    def _initialize_image_processor(self):
        """Initialize image processor"""
        self.image_processor = ImageProcessor()
    
    def process_image(
        self,
        image: Union[str, np.ndarray],
        extract_quality: bool = True,
        return_annotated: bool = True
    ) -> PipelineResult:
        """Process single image through complete pipeline"""
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # Step 1: Apple Detection
        detection_results = self.detector.detect(img)
        
        if not detection_results or not detection_results[0]['boxes']:
            # No apples detected
            processing_time = time.time() - start_time
            return PipelineResult(
                total_apples=0,
                active_tracks=0,
                unique_apples=0,
                quality_distribution={'good': 0, 'minor_defect': 0, 'major_defect': 0},
                quality_score=0.0,
                confidence_scores=[],
                processing_time=processing_time,
                annotated_image=img if return_annotated else None
            )
        
        detection_result = detection_results[0]
        
        # Step 2: Quality Assessment (if enabled)
        quality_results = []
        quality_distribution = {'good': 0, 'minor_defect': 0, 'major_defect': 0}
        quality_score = 0.0
        
        if extract_quality and detection_result['boxes']:
            # Extract apple patches
            apple_patches = self.image_processor.extract_patches(
                img, 
                detection_result['boxes'],
                patch_size=tuple(self.config.get('model', {}).get('quality_classification', {}).get('input_size', [224, 224]))
            )
            
            # Classify quality
            quality_results = self.quality_classifier.classify_apple_patches(apple_patches)
            
            # Calculate quality distribution and score
            for result in quality_results:
                quality_class = result['predicted_class']
                if quality_class in quality_distribution:
                    quality_distribution[quality_class] += 1
            
            quality_score = self._calculate_quality_score(quality_results)
        
        # Create annotated image
        annotated_image = None
        if return_annotated:
            annotated_image = self._create_annotated_image(img, detection_result, quality_results)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return PipelineResult(
            total_apples=detection_result['total_apples'],
            active_tracks=0,
            unique_apples=detection_result['total_apples'],
            quality_distribution=quality_distribution,
            quality_score=quality_score,
            confidence_scores=detection_result['confidence_scores'],
            processing_time=processing_time,
            annotated_image=annotated_image
        )
    
    def process_video(
        self,
        video_source: Union[str, int],
        extract_quality: bool = True,
        save_video: bool = False,
        output_path: Optional[str] = None
    ) -> List[PipelineResult]:
        """Process video through complete pipeline with tracking"""
        self.is_video_mode = True
        self.frame_count = 0
        results = []
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_result = self._process_video_frame(frame_rgb, extract_quality)
                results.append(frame_result)
                
                if video_writer and frame_result.annotated_image is not None:
                    annotated_frame = cv2.cvtColor(frame_result.annotated_image, cv2.COLOR_RGB2BGR)
                    video_writer.write(annotated_frame)
                
                self.frame_count += 1
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
        
        return results
    
    def _process_video_frame(self, frame: np.ndarray, extract_quality: bool = True) -> PipelineResult:
        """Process single video frame with tracking"""
        start_time = time.time()
        
        # Detection
        detection_results = self.detector.detect(frame)
        
        if not detection_results or not detection_results[0]['boxes']:
            tracks = []
        else:
            tracks = self.tracker.update(detection_results, frame)
        
        # Quality Assessment
        quality_results = []
        quality_distribution = {'good': 0, 'minor_defect': 0, 'major_defect': 0}
        quality_score = 0.0
        
        if extract_quality and tracks:
            track_boxes = [track['bbox'] for track in tracks]
            apple_patches = self.image_processor.extract_patches(frame, track_boxes)
            quality_results = self.quality_classifier.classify_apple_patches(apple_patches)
            
            for result in quality_results:
                quality_class = result['predicted_class']
                if quality_class in quality_distribution:
                    quality_distribution[quality_class] += 1
            
            quality_score = self._calculate_quality_score(quality_results)
        
        # Create annotated frame
        annotated_frame = self.tracker.visualize_tracks(frame, tracks, show_ids=True, show_trails=True)
        
        processing_time = time.time() - start_time
        
        return PipelineResult(
            total_apples=len(tracks),
            active_tracks=self.tracker.get_active_tracks_count(),
            unique_apples=self.tracker.get_unique_apple_count(),
            quality_distribution=quality_distribution,
            quality_score=quality_score,
            confidence_scores=[track['confidence'] for track in tracks],
            processing_time=processing_time,
            annotated_image=annotated_frame
        )
    
    def _calculate_quality_score(self, quality_results: List[Dict]) -> float:
        """Calculate overall quality score"""
        if not quality_results:
            return 0.0
        
        weights = {'good': 1.0, 'minor_defect': 0.7, 'major_defect': 0.3}
        total_score = 0.0
        
        for result in quality_results:
            class_name = result['predicted_class']
            confidence = result['confidence']
            weight = weights.get(class_name, 0.0)
            total_score += weight * confidence
        
        return total_score / len(quality_results)
    
    def _create_annotated_image(
        self,
        image: np.ndarray,
        detection_result: Dict,
        quality_results: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Create annotated image with detections and quality assessments"""
        annotated = image.copy()
        
        for i, box in enumerate(detection_result['boxes']):
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            confidence = detection_result['confidence_scores'][i]
            
            # Determine color based on quality
            color = (0, 255, 0)  # Default green
            quality_text = ""
            
            if quality_results and i < len(quality_results):
                quality_class = quality_results[i]['predicted_class']
                quality_conf = quality_results[i]['confidence']
                
                if quality_class == 'good':
                    color = (0, 255, 0)
                elif quality_class == 'minor_defect':
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 0)
                
                quality_text = f" | {quality_class}: {quality_conf:.2f}"
            
            # Draw bounding box and label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"Apple: {confidence:.2f}{quality_text}"
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add summary
        total_apples = detection_result['total_apples']
        cv2.putText(annotated, f"Total Apples: {total_apples}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def generate_report(self, results: List[PipelineResult]) -> Dict:
        """Generate comprehensive analysis report"""
        if not results:
            return {}
        
        total_images = len(results)
        total_apples = sum(r.total_apples for r in results)
        avg_apples = total_apples / total_images if total_images > 0 else 0
        
        # Quality distribution
        total_quality = {'good': 0, 'minor_defect': 0, 'major_defect': 0}
        for result in results:
            for quality_class, count in result.quality_distribution.items():
                total_quality[quality_class] += count
        
        # Performance metrics
        avg_time = np.mean([r.processing_time for r in results])
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'summary': {
                'total_images_processed': total_images,
                'total_apples_detected': total_apples,
                'average_apples_per_image': avg_apples,
                'average_processing_time_ms': avg_time * 1000,
                'average_fps': avg_fps
            },
            'quality_analysis': {
                'distribution': total_quality,
                'percentages': {
                    k: (v / total_apples * 100) if total_apples > 0 else 0
                    for k, v in total_quality.items()
                }
            }
        }
    
    def reset_pipeline(self):
        """Reset pipeline state"""
        self.tracker.reset()
        self.frame_count = 0
        self.processing_times.clear()
        self.is_video_mode = False


if __name__ == "__main__":
    # Example usage
    pipeline = ApplePipeline()
    
    # Process single image
    # result = pipeline.process_image('path/to/image.jpg')
    # print(f"Detected {result.total_apples} apples")