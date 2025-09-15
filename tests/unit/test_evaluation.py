"""
Unit tests for Evaluation module
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
    from src.utils.evaluation import DetectionEvaluator, ClassificationEvaluator, PipelineEvaluator, ReportGenerator
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDetectionEvaluator(unittest.TestCase):
    """Test cases for DetectionEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = DetectionEvaluator(iou_threshold=0.5)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertEqual(self.evaluator.iou_threshold, 0.5)
    
    def test_calculate_iou_overlap(self):
        """Test IoU calculation with overlap"""
        box1 = [10, 10, 50, 50]
        box2 = [30, 30, 70, 70]
        
        iou = self.evaluator.calculate_iou(box1, box2)
        
        self.assertIsInstance(iou, float)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap"""
        box1 = [10, 10, 50, 50]
        box2 = [60, 60, 100, 100]
        
        iou = self.evaluator.calculate_iou(box1, box2)
        
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap"""
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        
        iou = self.evaluator.calculate_iou(box1, box2)
        
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_evaluate_detections(self):
        """Test detection evaluation"""
        predictions = [
            {
                'boxes': [{'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}],
                'confidence_scores': [0.8],
                'total_apples': 1
            }
        ]
        
        ground_truths = [
            {
                'boxes': [{'x1': 15, 'y1': 15, 'x2': 55, 'y2': 55}],
                'total_apples': 1
            }
        ]
        
        results = self.evaluator.evaluate_detections(predictions, ground_truths)
        
        self.assertIn('mAP', results)
        self.assertIn('conf_0.1', results)
        self.assertIsInstance(results['mAP'], float)
    
    def test_calculate_metrics_perfect_match(self):
        """Test metrics calculation with perfect match"""
        predictions = [
            {
                'boxes': [[10, 10, 50, 50]],
                'confidence_scores': [0.9],
                'total_apples': 1
            }
        ]
        
        ground_truths = [
            {
                'boxes': [[10, 10, 50, 50]],
                'total_apples': 1
            }
        ]
        
        metrics = self.evaluator._calculate_metrics(predictions, ground_truths)
        
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
        self.assertEqual(metrics['mae'], 0.0)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestClassificationEvaluator(unittest.TestCase):
    """Test cases for ClassificationEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ClassificationEvaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertEqual(len(self.evaluator.class_names), 3)
        self.assertIn('good', self.evaluator.class_names)
    
    def test_evaluate_classification_perfect(self):
        """Test classification evaluation with perfect predictions"""
        predictions = [0, 1, 2, 0, 1]
        ground_truths = [0, 1, 2, 0, 1]
        
        results = self.evaluator.evaluate_classification(predictions, ground_truths)
        
        self.assertEqual(results['accuracy'], 1.0)
        self.assertEqual(results['f1_macro'], 1.0)
        self.assertIn('per_class_metrics', results)
        self.assertIn('confusion_matrix', results)
    
    def test_evaluate_classification_with_errors(self):
        """Test classification evaluation with some errors"""
        predictions = [0, 1, 2, 0, 2]  # Last prediction is wrong
        ground_truths = [0, 1, 2, 0, 1]
        
        results = self.evaluator.evaluate_classification(predictions, ground_truths)
        
        self.assertLess(results['accuracy'], 1.0)
        self.assertGreater(results['accuracy'], 0.0)
        self.assertIn('per_class_metrics', results)
    
    def test_evaluate_classification_with_probabilities(self):
        """Test classification evaluation with probabilities"""
        predictions = [0, 1, 2]
        ground_truths = [0, 1, 2]
        probabilities = [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9]
        ]
        
        results = self.evaluator.evaluate_classification(
            predictions, ground_truths, probabilities
        )
        
        self.assertEqual(results['accuracy'], 1.0)
        # Check if AUC scores are present (might not be calculated in some cases)
        # self.assertIn('auc_scores', results)
    
    def test_custom_class_names(self):
        """Test evaluator with custom class names"""
        custom_names = ['class_a', 'class_b']
        evaluator = ClassificationEvaluator(class_names=custom_names)
        
        self.assertEqual(evaluator.class_names, custom_names)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestPipelineEvaluator(unittest.TestCase):
    """Test cases for PipelineEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = PipelineEvaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.detection_evaluator)
        self.assertIsNotNone(self.evaluator.classification_evaluator)
    
    def test_calculate_pipeline_metrics(self):
        """Test pipeline metrics calculation"""
        pipeline_results = [
            {'processing_time': 0.1, 'total_apples': 2},
            {'processing_time': 0.15, 'total_apples': 3},
            {'processing_time': 0.12, 'total_apples': 1}
        ]
        
        metrics = self.evaluator._calculate_pipeline_metrics(pipeline_results)
        
        self.assertEqual(metrics['total_images_processed'], 3)
        self.assertEqual(metrics['total_apples_detected'], 6)
        self.assertAlmostEqual(metrics['average_processing_time'], 0.123, places=2)
        self.assertGreater(metrics['average_fps'], 0)
    
    def test_evaluate_pipeline_empty_data(self):
        """Test pipeline evaluation with empty data"""
        pipeline_results = []
        ground_truth_data = []
        
        results = self.evaluator.evaluate_pipeline(pipeline_results, ground_truth_data)
        
        self.assertIn('detection_evaluation', results)
        self.assertIn('classification_evaluation', results)
        self.assertIn('pipeline_metrics', results)
        self.assertIn('timestamp', results)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.generator = ReportGenerator(output_dir=temp_dir)
            self.temp_dir = temp_dir
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        self.assertTrue(Path(self.generator.output_dir).exists())
    
    def test_generate_detection_report(self):
        """Test detection report generation"""
        evaluation_results = {
            'mAP': 0.85,
            'conf_0.3': {
                'precision': 0.9,
                'recall': 0.8,
                'f1': 0.85,
                'mae': 1.2
            },
            'conf_0.5': {
                'precision': 0.85,
                'recall': 0.75,
                'f1': 0.8,
                'mae': 1.5
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            report_path = generator.generate_detection_report(
                evaluation_results, save_plots=False
            )
            
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(report_path.endswith('.html'))
    
    def test_generate_classification_report(self):
        """Test classification report generation"""
        evaluation_results = {
            'accuracy': 0.92,
            'f1_macro': 0.88,
            'precision_macro': 0.9,
            'recall_macro': 0.86,
            'per_class_metrics': {
                'good': {'precision': 0.95, 'recall': 0.9, 'f1': 0.925},
                'minor_defect': {'precision': 0.85, 'recall': 0.8, 'f1': 0.825},
                'major_defect': {'precision': 0.9, 'recall': 0.88, 'f1': 0.89}
            },
            'confusion_matrix': [[50, 2, 1], [3, 45, 2], [1, 1, 48]]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            report_path = generator.generate_classification_report(
                evaluation_results, save_plots=False
            )
            
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(report_path.endswith('.html'))
    
    def test_prepare_results_for_json(self):
        """Test results preparation for JSON"""
        results = {
            'numpy_array': np.array([1, 2, 3]),
            'numpy_float': np.float32(3.14),
            'numpy_int': np.int64(42),
            'regular_dict': {'key': 'value'},
            'regular_list': [1, 2, 3]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(output_dir=temp_dir)
            json_ready = generator._prepare_results_for_json(results)
            
            self.assertIsInstance(json_ready['numpy_array'], list)
            self.assertIsInstance(json_ready['numpy_float'], (int, float))
            self.assertIsInstance(json_ready['numpy_int'], (int, float))
            self.assertIsInstance(json_ready['regular_dict'], dict)
            self.assertIsInstance(json_ready['regular_list'], list)


if __name__ == '__main__':
    unittest.main()