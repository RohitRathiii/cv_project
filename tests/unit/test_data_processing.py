"""
Unit tests for Data Processing module
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
    from src.data.data_processing import ImageProcessor, DataManager
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestImageProcessor(unittest.TestCase):
    """Test cases for ImageProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor(input_size=(640, 640))
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.input_size, (640, 640))
    
    def test_letterbox_resize(self):
        """Test letterbox resize functionality"""
        resized, scale, padding = self.processor.letterbox_resize(
            self.test_image, (640, 640)
        )
        
        self.assertEqual(resized.shape, (640, 640, 3))
        self.assertIsInstance(scale, float)
        self.assertIsInstance(padding, tuple)
        self.assertEqual(len(padding), 2)
    
    def test_letterbox_resize_square_image(self):
        """Test letterbox resize with square image"""
        square_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        resized, scale, padding = self.processor.letterbox_resize(
            square_image, (640, 640)
        )
        
        self.assertEqual(resized.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640/400, places=2)
    
    def test_normalize_image(self):
        """Test image normalization"""
        normalized = self.processor.normalize_image(self.test_image)
        
        self.assertEqual(normalized.shape, self.test_image.shape)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(normalized.min(), -3)  # Roughly within expected range
        self.assertLessEqual(normalized.max(), 3)
    
    def test_normalize_image_custom_params(self):
        """Test image normalization with custom parameters"""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        normalized = self.processor.normalize_image(
            self.test_image, mean=mean, std=std
        )
        
        self.assertEqual(normalized.shape, self.test_image.shape)
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_extract_patches(self):
        """Test patch extraction"""
        bboxes = [
            {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100},
            {'x1': 200, 'y1': 200, 'x2': 300, 'y2': 300}
        ]
        
        patches = self.processor.extract_patches(
            self.test_image, bboxes, patch_size=(224, 224)
        )
        
        self.assertEqual(len(patches), 2)
        for patch in patches:
            self.assertEqual(patch.shape, (224, 224, 3))
            self.assertEqual(patch.dtype, np.uint8)
    
    def test_extract_patches_with_padding(self):
        """Test patch extraction with padding"""
        bboxes = [{'x1': 5, 'y1': 5, 'x2': 50, 'y2': 50}]
        
        patches = self.processor.extract_patches(
            self.test_image, bboxes, patch_size=(224, 224), padding=20
        )
        
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].shape, (224, 224, 3))
    
    def test_extract_patches_out_of_bounds(self):
        """Test patch extraction with out-of-bounds bbox"""
        bboxes = [{'x1': 600, 'y1': 450, 'x2': 700, 'y2': 500}]  # Beyond image bounds
        
        patches = self.processor.extract_patches(
            self.test_image, bboxes, patch_size=(224, 224)
        )
        
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].shape, (224, 224, 3))
    
    def test_create_mosaic(self):
        """Test mosaic creation"""
        # Create 4 test images
        images = [
            np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8),
            np.random.randint(0, 255, (350, 400, 3), dtype=np.uint8),
            np.random.randint(0, 255, (300, 350, 3), dtype=np.uint8),
            np.random.randint(0, 255, (380, 320, 3), dtype=np.uint8)
        ]
        
        # Create dummy labels (YOLO format)
        labels = [
            [[0, 0.5, 0.5, 0.2, 0.3]],  # class, x_center, y_center, width, height
            [[0, 0.3, 0.4, 0.15, 0.25]],
            [[0, 0.6, 0.6, 0.1, 0.2]],
            [[0, 0.4, 0.3, 0.2, 0.3]]
        ]
        
        mosaic, mosaic_labels = self.processor.create_mosaic(
            images, labels, mosaic_size=(640, 640)
        )
        
        self.assertEqual(mosaic.shape, (640, 640, 3))
        self.assertIsInstance(mosaic_labels, list)
        # Labels should be adjusted for mosaic coordinates
        for label in mosaic_labels:
            self.assertEqual(len(label), 5)  # class, x, y, w, h
    
    def test_create_mosaic_invalid_input(self):
        """Test mosaic creation with invalid input"""
        images = [self.test_image, self.test_image]  # Only 2 images instead of 4
        labels = [[], []]
        
        with self.assertRaises(ValueError):
            self.processor.create_mosaic(images, labels)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDataManager(unittest.TestCase):
    """Test cases for DataManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_manager = DataManager()
    
    def test_data_manager_initialization(self):
        """Test data manager initialization"""
        self.assertIsInstance(self.data_manager.config, dict)
    
    def test_data_manager_with_config(self):
        """Test data manager initialization with config"""
        config_data = """
        data:
          detection:
            dataset_path: "test/path"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            temp_path = f.name
        
        try:
            data_manager = DataManager(temp_path)
            self.assertIn('data', data_manager.config)
        finally:
            os.unlink(temp_path)
    
    def test_split_dataset(self):
        """Test dataset splitting functionality"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / 'source'
            output_dir = Path(temp_dir) / 'output'
            
            # Create source directory structure
            (source_dir / 'images').mkdir(parents=True)
            (source_dir / 'labels').mkdir(parents=True)
            
            # Create dummy image files
            for i in range(10):
                img_path = source_dir / 'images' / f'image_{i:03d}.jpg'
                img_path.write_bytes(b'dummy_image_data')
                
                label_path = source_dir / 'labels' / f'image_{i:03d}.txt'
                label_path.write_text('0 0.5 0.5 0.2 0.3\n')
            
            # Split dataset
            self.data_manager.split_dataset(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            )
            
            # Check if split directories were created
            self.assertTrue((output_dir / 'train' / 'images').exists())
            self.assertTrue((output_dir / 'val' / 'images').exists())
            self.assertTrue((output_dir / 'test' / 'images').exists())
            
            # Check file counts (approximately)
            train_count = len(list((output_dir / 'train' / 'images').glob('*.jpg')))
            val_count = len(list((output_dir / 'val' / 'images').glob('*.jpg')))
            test_count = len(list((output_dir / 'test' / 'images').glob('*.jpg')))
            
            self.assertEqual(train_count + val_count + test_count, 10)
    
    def test_create_dataset_yaml(self):
        """Test dataset YAML creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / 'dataset'
            yaml_path = Path(temp_dir) / 'dataset.yaml'
            
            self.data_manager.create_dataset_yaml(
                dataset_path=str(dataset_path),
                output_path=str(yaml_path),
                class_names=['apple', 'orange']
            )
            
            self.assertTrue(yaml_path.exists())
            
            # Read and verify content
            import yaml
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.assertEqual(config['nc'], 2)
            self.assertEqual(config['names'], ['apple', 'orange'])
    
    def test_analyze_dataset(self):
        """Test dataset analysis"""
        # Create temporary dataset structure
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / 'dataset'
            
            # Create train split
            train_images = dataset_dir / 'train' / 'images'
            train_labels = dataset_dir / 'train' / 'labels'
            train_images.mkdir(parents=True)
            train_labels.mkdir(parents=True)
            
            # Create dummy files
            for i in range(5):
                (train_images / f'img_{i}.jpg').write_bytes(b'dummy')
                (train_labels / f'img_{i}.txt').write_text('0 0.5 0.5 0.2 0.3\n')
            
            stats = self.data_manager.analyze_dataset(str(dataset_dir))
            
            self.assertIn('total_images', stats)
            self.assertIn('total_labels', stats)
            self.assertIn('splits', stats)
            self.assertEqual(stats['splits']['train']['images'], 5)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDataProcessingUtils(unittest.TestCase):
    """Test utility functions in data processing"""
    
    def test_get_detection_transforms(self):
        """Test detection transforms creation"""
        try:
            from src.data.data_processing import get_detection_transforms
            
            # Test with augmentation
            transform = get_detection_transforms(augment=True)
            self.assertIsNotNone(transform)
            
            # Test without augmentation
            transform = get_detection_transforms(augment=False)
            self.assertIsNotNone(transform)
        except ImportError:
            self.skipTest("Albumentations not available")
    
    def test_get_quality_transforms(self):
        """Test quality transforms creation"""
        try:
            from src.data.data_processing import get_quality_transforms
            
            # Test with augmentation
            transform = get_quality_transforms(augment=True, image_size=224)
            self.assertIsNotNone(transform)
            
            # Test without augmentation
            transform = get_quality_transforms(augment=False, image_size=224)
            self.assertIsNotNone(transform)
        except ImportError:
            self.skipTest("Albumentations not available")


if __name__ == '__main__':
    unittest.main()