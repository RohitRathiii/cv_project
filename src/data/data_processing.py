"""
Data Processing Utilities for Apple Detection Pipeline

This module provides utilities for image preprocessing, data augmentation,
dataset management, and data loading for both detection and classification tasks.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processing utilities for preprocessing and augmentation
    """
    
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize image processor
        
        Args:
            input_size: Target input size for processing
        """
        self.input_size = input_size
        
    def letterbox_resize(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int],
        fill_value: int = 114
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with letterboxing to maintain aspect ratio
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            fill_value: Fill value for padding
            
        Returns:
            Resized image, scale factor, and padding offsets
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create new image with padding
        new_image = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        new_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return new_image, scale, (pad_x, pad_y)
    
    def normalize_image(
        self, 
        image: np.ndarray, 
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> np.ndarray:
        """
        Normalize image with ImageNet statistics
        
        Args:
            image: Input image (0-255 range)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Normalized image
        """
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    def extract_patches(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        patch_size: Tuple[int, int] = (224, 224),
        padding: int = 10
    ) -> List[np.ndarray]:
        """
        Extract image patches from bounding boxes
        
        Args:
            image: Source image
            bboxes: List of bounding box dictionaries
            patch_size: Size of extracted patches
            padding: Padding around bounding boxes
            
        Returns:
            List of extracted image patches
        """
        patches = []
        h, w = image.shape[:2]
        
        for bbox in bboxes:
            # Extract coordinates with padding
            x1 = max(0, int(bbox['x1']) - padding)
            y1 = max(0, int(bbox['y1']) - padding)
            x2 = min(w, int(bbox['x2']) + padding)
            y2 = min(h, int(bbox['y2']) + padding)
            
            # Extract patch
            patch = image[y1:y2, x1:x2]
            
            if patch.size > 0:
                # Resize to target size
                patch_resized = cv2.resize(patch, patch_size, interpolation=cv2.INTER_LINEAR)
                patches.append(patch_resized)
            else:
                # Create empty patch if extraction failed
                empty_patch = np.zeros((*patch_size, 3), dtype=np.uint8)
                patches.append(empty_patch)
        
        return patches
    
    def create_mosaic(
        self,
        images: List[np.ndarray],
        labels: List[List[List[float]]],
        mosaic_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Create mosaic augmentation from 4 images
        
        Args:
            images: List of 4 input images
            labels: List of labels for each image (YOLO format)
            mosaic_size: Size of output mosaic
            
        Returns:
            Mosaic image and updated labels
        """
        if len(images) != 4:
            raise ValueError("Mosaic requires exactly 4 images")
        
        mosaic_w, mosaic_h = mosaic_size
        
        # Create empty mosaic
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_labels = []
        
        # Define quadrant positions
        positions = [
            (0, 0, mosaic_w // 2, mosaic_h // 2),  # Top-left
            (mosaic_w // 2, 0, mosaic_w, mosaic_h // 2),  # Top-right
            (0, mosaic_h // 2, mosaic_w // 2, mosaic_h),  # Bottom-left
            (mosaic_w // 2, mosaic_h // 2, mosaic_w, mosaic_h)  # Bottom-right
        ]
        
        for i, (image, image_labels) in enumerate(zip(images, labels)):
            # Get quadrant position
            x1, y1, x2, y2 = positions[i]
            quad_w, quad_h = x2 - x1, y2 - y1
            
            # Resize image to fit quadrant
            resized = cv2.resize(image, (quad_w, quad_h))
            
            # Place in mosaic
            mosaic[y1:y2, x1:x2] = resized
            
            # Update labels for new position and scale
            h_orig, w_orig = image.shape[:2]
            scale_x = quad_w / w_orig
            scale_y = quad_h / h_orig
            
            for label in image_labels:
                if len(label) >= 5:  # class, x_center, y_center, width, height
                    class_id, x_center, y_center, width, height = label[:5]
                    
                    # Scale and translate to mosaic coordinates
                    new_x_center = (x_center * w_orig * scale_x + x1) / mosaic_w
                    new_y_center = (y_center * h_orig * scale_y + y1) / mosaic_h
                    new_width = width * scale_x
                    new_height = height * scale_y
                    
                    # Only keep labels that are still within bounds
                    if (0 < new_x_center < 1 and 0 < new_y_center < 1 and
                        new_width > 0 and new_height > 0):
                        mosaic_labels.append([
                            class_id, new_x_center, new_y_center, new_width, new_height
                        ])
        
        return mosaic, mosaic_labels


class DetectionDataset(Dataset):
    """
    Dataset class for apple detection training with YOLO format
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        mosaic_prob: float = 0.5
    ):
        """
        Initialize detection dataset
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            image_size: Target image size
            augment: Whether to apply augmentations
            mosaic_prob: Probability of mosaic augmentation
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        
        # Get image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        self.image_files = sorted(self.image_files)
        
        # Setup augmentations
        self.processor = ImageProcessor(image_size)
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Blur(blur_limit=3, p=0.1),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = None
        
        logger.info(f"Loaded {len(self.image_files)} images for detection dataset")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        labels = self._load_labels(label_path)
        
        # Apply mosaic augmentation
        if self.augment and np.random.random() < self.mosaic_prob and len(self.image_files) >= 4:
            # Get 3 additional random images
            other_indices = np.random.choice(
                [i for i in range(len(self.image_files)) if i != idx], 
                size=3, 
                replace=False
            )
            
            other_images = []
            other_labels = []
            
            for other_idx in other_indices:
                other_image_path = self.image_files[other_idx]
                other_image = cv2.imread(str(other_image_path))
                other_image = cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB)
                other_images.append(other_image)
                
                other_label_path = self.labels_dir / f"{other_image_path.stem}.txt"
                other_labels.append(self._load_labels(other_label_path))
            
            # Create mosaic
            all_images = [image] + other_images
            all_labels = [labels] + other_labels
            image, labels = self.processor.create_mosaic(all_images, all_labels, self.image_size)
        
        # Apply other augmentations
        if self.transform and labels:
            # Convert labels to albumentations format
            bboxes = []
            class_labels = []
            
            for label in labels:
                if len(label) >= 5:
                    class_id, x_center, y_center, width, height = label[:5]
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(int(class_id))
            
            if bboxes:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                
                # Convert back to YOLO format
                labels = []
                for bbox, class_id in zip(transformed['bboxes'], transformed['class_labels']):
                    labels.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Resize image
        image, scale, padding = self.processor.letterbox_resize(image, self.image_size)
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Prepare labels tensor
        if labels:
            labels_tensor = torch.zeros((len(labels), 6))  # batch_idx, class, x, y, w, h
            for i, label in enumerate(labels):
                if len(label) >= 5:
                    labels_tensor[i, 1:] = torch.tensor(label[:5])
        else:
            labels_tensor = torch.zeros((0, 6))
        
        return {
            'image': image,
            'labels': labels_tensor,
            'image_path': str(image_path)
        }
    
    def _load_labels(self, label_path: Path) -> List[List[float]]:
        """Load YOLO format labels"""
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts])
        return labels


class QualityDataset(Dataset):
    """
    Dataset class for apple quality classification
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
        class_names: List[str] = None
    ):
        """
        Initialize quality dataset
        
        Args:
            root_dir: Root directory with class subdirectories
            transform: Transforms to apply to images
            class_names: List of class names (inferred from directories if None)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get class names from subdirectories
        if class_names is None:
            self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Collect all samples
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        logger.info(f"Loaded {len(self.samples)} samples for quality dataset")
        logger.info(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            if hasattr(self.transform, '__call__'):
                # Albumentations transform
                if 'albumentations' in str(type(self.transform)):
                    transformed = self.transform(image=image)
                    image = transformed['image']
                else:
                    # PyTorch transforms
                    image = self.transform(image)
        
        return image, label


class DataManager:
    """
    Data management utilities for dataset download, preparation, and splitting
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    def download_roboflow_dataset(
        self,
        workspace: str,
        project: str,
        version: int,
        api_key: str,
        format: str = 'yolov8',
        location: str = 'datasets'
    ) -> str:
        """
        Download dataset from Roboflow
        
        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version
            api_key: Roboflow API key
            format: Dataset format
            location: Download location
            
        Returns:
            Path to downloaded dataset
        """
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download(format, location=location)
            
            logger.info(f"Dataset downloaded to {dataset.location}")
            return dataset.location
            
        except ImportError:
            logger.error("Roboflow not installed. Install with: pip install roboflow")
            raise
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def split_dataset(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Split dataset into train/val/test sets
        
        Args:
            source_dir: Source directory with images and labels
            output_dir: Output directory for split dataset
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list((source_path / 'images').glob(ext)))
        
        # Shuffle files
        image_files = sorted(image_files)
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_file in files:
                # Copy image
                dst_img = output_path / split / 'images' / img_file.name
                dst_img.write_bytes(img_file.read_bytes())
                
                # Copy label if exists
                label_file = source_path / 'labels' / f"{img_file.stem}.txt"
                if label_file.exists():
                    dst_label = output_path / split / 'labels' / f"{img_file.stem}.txt"
                    dst_label.write_text(label_file.read_text())
        
        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train_files)} files")
        logger.info(f"  Val: {len(val_files)} files")
        logger.info(f"  Test: {len(test_files)} files")
    
    def create_dataset_yaml(
        self,
        dataset_path: str,
        output_path: str,
        class_names: List[str] = ['apple']
    ):
        """
        Create YOLO dataset configuration file
        
        Args:
            dataset_path: Path to dataset directory
            output_path: Output path for YAML file
            class_names: List of class names
        """
        config = {
            'path': dataset_path,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Dataset YAML created at {output_path}")
    
    def analyze_dataset(self, dataset_path: str) -> Dict:
        """
        Analyze dataset statistics
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_path = Path(dataset_path)
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'splits': {},
            'class_distribution': {},
            'image_sizes': [],
            'bbox_sizes': []
        }
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                images_path = split_path / 'images'
                labels_path = split_path / 'labels'
                
                # Count files
                image_files = list(images_path.glob('*'))
                label_files = list(labels_path.glob('*.txt'))
                
                stats['splits'][split] = {
                    'images': len(image_files),
                    'labels': len(label_files)
                }
                
                stats['total_images'] += len(image_files)
                stats['total_labels'] += len(label_files)
                
                # Analyze image sizes
                for img_file in image_files[:100]:  # Sample first 100 images
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            stats['image_sizes'].append((w, h))
                    except:
                        continue
                
                # Analyze labels
                for label_file in label_files:
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    width, height = float(parts[3]), float(parts[4])
                                    
                                    # Update class distribution
                                    if class_id not in stats['class_distribution']:
                                        stats['class_distribution'][class_id] = 0
                                    stats['class_distribution'][class_id] += 1
                                    
                                    # Update bbox sizes
                                    stats['bbox_sizes'].append((width, height))
                    except:
                        continue
        
        logger.info(f"Dataset analysis completed:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Total labels: {stats['total_labels']}")
        logger.info(f"  Class distribution: {stats['class_distribution']}")
        
        return stats


def get_detection_transforms(augment: bool = True, image_size: int = 640):
    """
    Get transforms for detection dataset
    
    Args:
        augment: Whether to include augmentations
        image_size: Target image size
        
    Returns:
        Transform pipeline
    """
    if augment:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(p=0.1),
            A.ToGray(p=0.01),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2()
        ])


def get_quality_transforms(augment: bool = True, image_size: int = 224):
    """
    Get transforms for quality classification dataset
    
    Args:
        augment: Whether to include augmentations
        image_size: Target image size
        
    Returns:
        Transform pipeline
    """
    if augment:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


if __name__ == "__main__":
    # Example usage
    data_manager = DataManager()
    
    # Example dataset analysis
    # stats = data_manager.analyze_dataset('datasets/detection')
    # print(f"Dataset statistics: {stats}")
    
    # Example dataset creation
    # train_dataset = DetectionDataset(
    #     images_dir='datasets/detection/train/images',
    #     labels_dir='datasets/detection/train/labels',
    #     augment=True
    # )
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)