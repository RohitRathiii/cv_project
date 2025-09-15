"""
Unit tests for Apple Detector module
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models.apple_detector import AppleDetector
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestAppleDetector(unittest.TestCase):
    """Test cases for AppleDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    @patch('src.models.apple_detector.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = AppleDetector(device='cpu', conf_threshold=0.5)
        
        self.assertEqual(detector.device, 'cpu')
        self.assertEqual(detector.conf_threshold, 0.5)
        mock_yolo.assert_called_once()
    
    @patch('src.models.apple_detector.YOLO')
    def test_device_setup_auto(self, mock_yolo):
        """Test automatic device setup"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            detector = AppleDetector(device='auto')
            self.assertEqual(detector.device, 'cpu')
    
    @patch('src.models.apple_detector.YOLO')
    def test_detect_with_results(self, mock_yolo):
        """Test detection with mock results"""
        # Setup mock
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        
        # Mock boxes data
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8, 0.9])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 0])
        
        mock_result.boxes = mock_boxes
        mock_result.orig_shape = (480, 640)
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = AppleDetector(device='cpu')
        results = detector.detect(self.test_image)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['total_apples'], 2)
        self.assertEqual(len(results[0]['boxes']), 2)
        self.assertEqual(len(results[0]['confidence_scores']), 2)
    
    @patch('src.models.apple_detector.YOLO')
    def test_detect_no_results(self, mock_yolo):
        """Test detection with no results"""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_result.orig_shape = (480, 640)
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = AppleDetector(device='cpu')
        results = detector.detect(self.test_image)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['total_apples'], 0)
        self.assertEqual(len(results[0]['boxes']), 0)
    
    @patch('src.models.apple_detector.YOLO')
    def test_annotate_image(self, mock_yolo):
        """Test image annotation"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = AppleDetector(device='cpu')
        
        detection_result = {
            'boxes': [
                {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50},
                {'x1': 60, 'y1': 60, 'x2': 100, 'y2': 100}
            ],
            'confidence_scores': [0.8, 0.9],
            'total_apples': 2
        }
        
        annotated = detector.annotate_image(self.test_image, detection_result)
        
        self.assertEqual(annotated.shape, self.test_image.shape)
        self.assertIsInstance(annotated, np.ndarray)
    
    @patch('src.models.apple_detector.YOLO')
    def test_performance_stats(self, mock_yolo):
        """Test performance statistics tracking"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = AppleDetector(device='cpu')
        
        # Simulate some inference times
        detector.inference_times = [0.1, 0.2, 0.15]
        
        stats = detector.get_performance_stats()
        
        self.assertIn('avg_inference_time', stats)
        self.assertIn('fps', stats)
        self.assertEqual(stats['total_inferences'], 3)
        self.assertAlmostEqual(stats['avg_inference_time'], 0.15, places=2)


class TestAppleDetectorUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_load_config(self):
        """Test configuration loading"""
        # Create temporary config file
        config_data = """
        model:
          detection:
            confidence_threshold: 0.3
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            temp_path = f.name
        
        try:
            if IMPORTS_AVAILABLE:
                from src.models.apple_detector import load_config
                config = load_config(temp_path)
                self.assertIn('model', config)
                self.assertEqual(config['model']['detection']['confidence_threshold'], 0.3)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()