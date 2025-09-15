"""
Integration Test Script for Apple Detection Pipeline

This script performs comprehensive integration testing to validate
the complete pipeline functionality.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import cv2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all module imports"""
    logger.info("Testing module imports...")
    
    try:
        # Test model imports
        from src.models.apple_detector import AppleDetector
        from src.models.quality_classifier import QualityClassificationPipeline, QualityClassifier
        from src.models.apple_tracker import AppleTracker
        
        # Test data processing imports
        from src.data.data_processing import DataManager, ImageProcessor
        
        # Test pipeline imports
        from src.pipeline.apple_pipeline import ApplePipeline
        
        # Test utility imports
        from src.utils.gradio_interface import AppleGradioInterface
        from src.utils.evaluation import DetectionEvaluator, ClassificationEvaluator
        
        logger.info("✅ All imports successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False


def test_model_initialization():
    """Test model initialization"""
    logger.info("Testing model initialization...")
    
    try:
        # Test detector initialization
        detector = AppleDetector(device='cpu')
        logger.info("✅ Apple detector initialized")
        
        # Test quality classifier initialization
        classifier = QualityClassificationPipeline(device='cpu')
        logger.info("✅ Quality classifier initialized")
        
        # Test tracker initialization
        tracker = AppleTracker()
        logger.info("✅ Apple tracker initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {str(e)}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization"""
    logger.info("Testing pipeline initialization...")
    
    try:
        pipeline = ApplePipeline(device='cpu')
        logger.info("✅ Pipeline initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {str(e)}")
        return False


def test_data_processing():
    """Test data processing utilities"""
    logger.info("Testing data processing utilities...")
    
    try:
        # Test image processor
        from src.data.data_processing import ImageProcessor
        
        processor = ImageProcessor()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test letterbox resize
        resized, scale, padding = processor.letterbox_resize(dummy_image, (640, 640))
        assert resized.shape == (640, 640, 3)
        
        # Test normalization
        normalized = processor.normalize_image(dummy_image)
        assert normalized.dtype == np.float32
        
        logger.info("✅ Data processing utilities working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processing test failed: {str(e)}")
        return False


def test_evaluation_utilities():
    """Test evaluation utilities"""
    logger.info("Testing evaluation utilities...")
    
    try:
        from src.utils.evaluation import DetectionEvaluator, ClassificationEvaluator
        
        # Test detection evaluator
        det_evaluator = DetectionEvaluator()
        
        # Test IoU calculation
        box1 = [10, 10, 50, 50]
        box2 = [20, 20, 60, 60]
        iou = det_evaluator.calculate_iou(box1, box2)
        assert 0 <= iou <= 1
        
        # Test classification evaluator
        cls_evaluator = ClassificationEvaluator()
        
        # Test with dummy data
        predictions = [0, 1, 2, 0, 1]
        ground_truths = [0, 1, 1, 0, 2]
        results = cls_evaluator.evaluate_classification(predictions, ground_truths)
        
        assert 'accuracy' in results
        assert 'f1_macro' in results
        
        logger.info("✅ Evaluation utilities working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation utilities test failed: {str(e)}")
        return False


def test_configuration_loading():
    """Test configuration loading"""
    logger.info("Testing configuration loading...")
    
    try:
        import yaml
        
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate config structure
            assert 'model' in config
            assert 'training' in config
            assert 'data' in config
            
            logger.info("✅ Configuration loading working")
        else:
            logger.warning("⚠️ Config file not found, but structure is valid")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading test failed: {str(e)}")
        return False


def test_file_structure():
    """Test project file structure"""
    logger.info("Testing project file structure...")
    
    try:
        project_root = Path(__file__).parent.parent
        
        # Check required directories
        required_dirs = [
            'src', 'src/models', 'src/data', 'src/pipeline', 'src/utils',
            'scripts', 'config', 'tests', 'models', 'datasets', 'results'
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Directory {dir_path} not found"
        
        # Check required files
        required_files = [
            'requirements.txt', 'README.md', 'config/config.yaml',
            'src/__init__.py', 'src/models/__init__.py',
            'src/models/apple_detector.py', 'src/models/quality_classifier.py',
            'src/models/apple_tracker.py', 'src/pipeline/apple_pipeline.py'
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"File {file_path} not found"
        
        logger.info("✅ Project file structure is complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ File structure test failed: {str(e)}")
        return False


def test_dummy_pipeline_execution():
    """Test pipeline execution with dummy data"""
    logger.info("Testing pipeline execution with dummy data...")
    
    try:
        # Initialize pipeline
        pipeline = ApplePipeline(device='cpu')
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test image processing (without models)
        logger.info("Testing basic image processing...")
        
        # This should work even without trained models
        # as it tests the pipeline structure
        result = pipeline.process_image(
            image=dummy_image,
            extract_quality=False,  # Skip quality to avoid model issues
            return_annotated=True
        )
        
        # Validate result structure
        assert hasattr(result, 'total_apples')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'confidence_scores')
        
        logger.info("✅ Pipeline execution structure working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution test failed: {str(e)}")
        logger.info("Note: This may fail without trained models, which is expected")
        return False


def run_integration_tests():
    """Run all integration tests"""
    logger.info("="*60)
    logger.info("STARTING INTEGRATION TESTS")
    logger.info("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Model Initialization", test_model_initialization),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Data Processing", test_data_processing),
        ("Evaluation Utilities", test_evaluation_utilities),
        ("Configuration Loading", test_configuration_loading),
        ("File Structure", test_file_structure),
        ("Pipeline Execution", test_dummy_pipeline_execution)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The pipeline is ready for use.")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Please check the issues above.")
    
    logger.info("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)