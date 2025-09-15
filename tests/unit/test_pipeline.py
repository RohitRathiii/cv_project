"""
Unit tests for Apple Pipeline module
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.pipeline.apple_pipeline import ApplePipeline, PipelineResult
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestPipelineResult(unittest.TestCase):
    """Test cases for PipelineResult dataclass"""
    
    def test_pipeline_result_creation(self):
        """Test PipelineResult creation"""
        result = PipelineResult(
            total_apples=5,
            active_tracks=3,
            unique_apples=5,
            quality_distribution={'good': 3, 'minor_defect': 1, 'major_defect': 1},
            quality_score=0.85,
            confidence_scores=[0.8, 0.9, 0.7, 0.85, 0.95],
            processing_time=0.15
        )
        
        self.assertEqual(result.total_apples, 5)
        self.assertEqual(result.active_tracks, 3)
        self.assertEqual(result.unique_apples, 5)
        self.assertEqual(result.quality_score, 0.85)
        self.assertEqual(len(result.confidence_scores), 5)
        self.assertEqual(result.processing_time, 0.15)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestApplePipeline(unittest.TestCase):
    """Test cases for ApplePipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_pipeline_initialization(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test pipeline initialization"""
        pipeline = ApplePipeline(device='cpu')
        
        self.assertEqual(pipeline.device, 'cpu')
        self.assertFalse(pipeline.is_video_mode)
        self.assertEqual(pipeline.frame_count, 0)
        mock_detector.assert_called_once()
        mock_quality.assert_called_once()
        mock_tracker.assert_called_once()
        mock_processor.assert_called_once()
    
    def test_device_setup_auto(self):
        """Test automatic device setup"""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('src.pipeline.apple_pipeline.AppleDetector'), \
                 patch('src.pipeline.apple_pipeline.QualityClassificationPipeline'), \
                 patch('src.pipeline.apple_pipeline.AppleTracker'), \
                 patch('src.pipeline.apple_pipeline.ImageProcessor'):
                
                pipeline = ApplePipeline(device='auto')
                self.assertEqual(pipeline.device, 'cpu')
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_load_config_with_file(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test configuration loading with file"""
        config_data = """
        model:
          detection:
            confidence_threshold: 0.5
          quality_classification:
            input_size: [224, 224]
          tracking:
            max_age: 30
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            temp_path = f.name
        
        try:
            pipeline = ApplePipeline(config_path=temp_path, device='cpu')
            self.assertEqual(pipeline.config['model']['detection']['confidence_threshold'], 0.5)
        finally:
            os.unlink(temp_path)
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_process_image_no_detections(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test image processing with no detections"""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []
        mock_detector.return_value = mock_detector_instance
        
        mock_quality_instance = Mock()
        mock_quality.return_value = mock_quality_instance
        
        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance
        
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        
        pipeline = ApplePipeline(device='cpu')
        result = pipeline.process_image(self.test_image)
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.total_apples, 0)
        self.assertEqual(result.unique_apples, 0)
        self.assertEqual(result.quality_score, 0.0)
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_process_image_with_detections(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test image processing with detections"""
        # Setup mocks
        mock_detection_result = {
            'boxes': [
                {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50},
                {'x1': 60, 'y1': 60, 'x2': 100, 'y2': 100}
            ],
            'confidence_scores': [0.8, 0.9],
            'total_apples': 2
        }
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = [mock_detection_result]
        mock_detector.return_value = mock_detector_instance
        
        mock_quality_instance = Mock()
        mock_quality_results = [
            {'predicted_class': 'good', 'confidence': 0.9},
            {'predicted_class': 'minor_defect', 'confidence': 0.8}
        ]
        mock_quality_instance.classify_apple_patches.return_value = mock_quality_results
        mock_quality.return_value = mock_quality_instance
        
        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance
        
        mock_processor_instance = Mock()
        mock_processor_instance.extract_patches.return_value = [self.test_image[:224, :224], self.test_image[:224, :224]]
        mock_processor.return_value = mock_processor_instance
        
        pipeline = ApplePipeline(device='cpu')
        result = pipeline.process_image(self.test_image, extract_quality=True)
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.total_apples, 2)
        self.assertEqual(result.quality_distribution['good'], 1)
        self.assertEqual(result.quality_distribution['minor_defect'], 1)
        self.assertGreater(result.quality_score, 0)
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_calculate_quality_score(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test quality score calculation"""
        mock_detector.return_value = Mock()
        mock_quality.return_value = Mock()
        mock_tracker.return_value = Mock()
        mock_processor.return_value = Mock()
        
        pipeline = ApplePipeline(device='cpu')
        
        quality_results = [
            {'predicted_class': 'good', 'confidence': 0.9},
            {'predicted_class': 'minor_defect', 'confidence': 0.8},
            {'predicted_class': 'major_defect', 'confidence': 0.7}
        ]
        
        score = pipeline._calculate_quality_score(quality_results)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_generate_report(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test report generation"""
        mock_detector.return_value = Mock()
        mock_quality.return_value = Mock()
        mock_tracker.return_value = Mock()
        mock_processor.return_value = Mock()
        
        pipeline = ApplePipeline(device='cpu')
        
        # Create sample results
        results = [
            PipelineResult(
                total_apples=2,
                active_tracks=2,
                unique_apples=2,
                quality_distribution={'good': 1, 'minor_defect': 1, 'major_defect': 0},
                quality_score=0.85,
                confidence_scores=[0.8, 0.9],
                processing_time=0.15
            ),
            PipelineResult(
                total_apples=3,
                active_tracks=3,
                unique_apples=3,
                quality_distribution={'good': 2, 'minor_defect': 0, 'major_defect': 1},
                quality_score=0.77,
                confidence_scores=[0.7, 0.85, 0.95],
                processing_time=0.18
            )
        ]
        
        report = pipeline.generate_report(results)
        
        self.assertIn('summary', report)
        self.assertIn('quality_analysis', report)
        self.assertEqual(report['summary']['total_images_processed'], 2)
        self.assertEqual(report['summary']['total_apples_detected'], 5)
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_reset_pipeline(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test pipeline reset"""
        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance
        
        mock_detector.return_value = Mock()
        mock_quality.return_value = Mock()
        mock_processor.return_value = Mock()
        
        pipeline = ApplePipeline(device='cpu')
        
        # Simulate some processing
        pipeline.frame_count = 5
        pipeline.processing_times = [0.1, 0.2, 0.15]
        pipeline.is_video_mode = True
        
        # Reset pipeline
        pipeline.reset_pipeline()
        
        self.assertEqual(pipeline.frame_count, 0)
        self.assertEqual(len(pipeline.processing_times), 0)
        self.assertFalse(pipeline.is_video_mode)
        mock_tracker_instance.reset.assert_called_once()


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestPipelineUtils(unittest.TestCase):
    """Test pipeline utility functions"""
    
    @patch('src.pipeline.apple_pipeline.AppleDetector')
    @patch('src.pipeline.apple_pipeline.QualityClassificationPipeline')
    @patch('src.pipeline.apple_pipeline.AppleTracker')
    @patch('src.pipeline.apple_pipeline.ImageProcessor')
    def test_create_annotated_image(self, mock_processor, mock_tracker, mock_quality, mock_detector):
        """Test annotated image creation"""
        mock_detector.return_value = Mock()
        mock_quality.return_value = Mock()
        mock_tracker.return_value = Mock()
        mock_processor.return_value = Mock()
        
        pipeline = ApplePipeline(device='cpu')
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection_result = {
            'boxes': [{'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}],
            'confidence_scores': [0.8],
            'total_apples': 1
        }
        quality_results = [{'predicted_class': 'good', 'confidence': 0.9}]
        
        annotated = pipeline._create_annotated_image(test_image, detection_result, quality_results)
        
        self.assertEqual(annotated.shape, test_image.shape)
        self.assertIsInstance(annotated, np.ndarray)


if __name__ == '__main__':
    unittest.main()