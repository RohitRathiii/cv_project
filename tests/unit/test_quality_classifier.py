"""
Unit tests for Quality Classifier module
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models.quality_classifier import QualityClassifier, QualityClassificationPipeline
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestQualityClassifier(unittest.TestCase):
    """Test cases for QualityClassifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = QualityClassifier(num_classes=3, pretrained=False)
        self.test_input = torch.randn(1, 3, 224, 224)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.num_classes, 3)
        self.assertEqual(len(self.model.class_names), 3)
        self.assertIn('good', self.model.class_names)
        self.assertIn('minor_defect', self.model.class_names)
        self.assertIn('major_defect', self.model.class_names)
    
    def test_forward_pass(self):
        """Test forward pass"""
        output = self.model(self.test_input)
        
        self.assertEqual(output.shape, (1, 3))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_model_size_calculation(self):
        """Test model size calculation"""
        size_mb = self.model.get_model_size()
        
        self.assertIsInstance(size_mb, float)
        self.assertGreater(size_mb, 0)
    
    def test_custom_num_classes(self):
        """Test custom number of classes"""
        model = QualityClassifier(num_classes=5, pretrained=False)
        output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, 5))


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestQualityClassificationPipeline(unittest.TestCase):
    """Test cases for QualityClassificationPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = QualityClassifier(num_classes=3, pretrained=False)
        self.pipeline = QualityClassificationPipeline(model=self.model, device='cpu')
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.device, 'cpu')
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.transform_val)
    
    def test_device_setup_auto(self):
        """Test automatic device setup"""
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = QualityClassificationPipeline(device='auto')
            self.assertEqual(pipeline.device, 'cpu')
    
    def test_preprocess_image_numpy(self):
        """Test image preprocessing with numpy array"""
        tensor = self.pipeline.preprocess_image(self.test_image)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
    
    def test_preprocess_image_path(self):
        """Test image preprocessing with file path"""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save test image
            import cv2
            cv2.imwrite(temp_path, cv2.cvtColor(self.test_image, cv2.COLOR_RGB2BGR))
            
            tensor = self.pipeline.preprocess_image(temp_path)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, (1, 3, 224, 224))
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_classify_single_image(self):
        """Test single image classification"""
        predicted_class, confidence = self.pipeline.classify_single(self.test_image)
        
        self.assertIn(predicted_class, self.model.class_names)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_classify_single_with_probabilities(self):
        """Test single image classification with probabilities"""
        predicted_class, confidence, probabilities = self.pipeline.classify_single(
            self.test_image, return_probabilities=True
        )
        
        self.assertIn(predicted_class, self.model.class_names)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
    
    def test_classify_apple_patches(self):
        """Test apple patches classification"""
        patches = [self.test_image, self.test_image, self.test_image]
        results = self.pipeline.classify_apple_patches(patches)
        
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result['patch_id'], i)
            self.assertIn('predicted_class', result)
            self.assertIn('confidence', result)
            self.assertIn('probabilities', result)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        predictions = [
            {'predicted_class': 'good', 'confidence': 0.9},
            {'predicted_class': 'minor_defect', 'confidence': 0.8},
            {'predicted_class': 'major_defect', 'confidence': 0.7}
        ]
        
        from src.models.quality_classifier import calculate_quality_score
        score = calculate_quality_score(predictions)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        # Simulate some classifications
        self.pipeline.inference_times = [0.05, 0.06, 0.04]
        
        stats = self.pipeline.get_performance_stats()
        
        self.assertIn('avg_inference_time', stats)
        self.assertIn('fps', stats)
        self.assertEqual(stats['total_inferences'], 3)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestQualityClassifierTraining(unittest.TestCase):
    """Test training-related functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = QualityClassifier(num_classes=3, pretrained=False)
        self.pipeline = QualityClassificationPipeline(model=self.model, device='cpu')
    
    @patch('torch.utils.data.DataLoader')
    def test_validation_method(self, mock_dataloader):
        """Test validation method"""
        # Mock data loader
        mock_data = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
        mock_dataloader.return_value = mock_data
        
        mock_criterion = nn.CrossEntropyLoss()
        
        val_loss, accuracy = self.pipeline._validate(mock_dataloader, mock_criterion)
        
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)
    
    def test_training_history_initialization(self):
        """Test training history initialization"""
        self.assertIn('train_loss', self.pipeline.training_history)
        self.assertIn('val_loss', self.pipeline.training_history)
        self.assertIn('val_accuracy', self.pipeline.training_history)
        self.assertIsInstance(self.pipeline.training_history['train_loss'], list)


if __name__ == '__main__':
    unittest.main()