"""
MinneApple Dataset Downloader and Setup Script

This script helps download and organize the MinneApple dataset in the correct
format for your apple detection pipeline.
"""

import os
import sys
import requests
import logging
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List
import json
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinneAppleDatasetDownloader:
    """
    Download and setup MinneApple dataset
    """

    def __init__(self, project_root: str = None):
        """
        Initialize the downloader

        Args:
            project_root: Root directory of the project
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent

        self.project_root = Path(project_root)
        self.datasets_dir = self.project_root / "datasets"
        self.detection_dir = self.datasets_dir / "detection"
        self.quality_dir = self.datasets_dir / "quality"

        # Create directory structure
        self.setup_directory_structure()

        # Dataset URLs and information
        self.minneapple_info = {
            "name": "MinneApple",
            "description": "Benchmark dataset for apple detection and segmentation",
            "total_images": 1000,
            "total_instances": 41000,
            "official_url": "https://conservancy.umn.edu/handle/11299/206575",
            "roboflow_url": "https://universe.roboflow.com/dissertation-bqltf/minneapple-u6uyg/dataset/1",
            "github_url": "https://github.com/nicolaihaeni/MinneApple"
        }

        logger.info("MinneApple Dataset Downloader initialized")

    def setup_directory_structure(self):
        """Create the required directory structure"""

        # Detection dataset structure (YOLO format)
        detection_dirs = [
            self.detection_dir / "images" / "train",
            self.detection_dir / "images" / "val",
            self.detection_dir / "images" / "test",
            self.detection_dir / "labels" / "train",
            self.detection_dir / "labels" / "val",
            self.detection_dir / "labels" / "test"
        ]

        # Quality dataset structure (Classification format)
        quality_dirs = [
            self.quality_dir / "train" / "good",
            self.quality_dir / "train" / "minor_defect",
            self.quality_dir / "train" / "major_defect",
            self.quality_dir / "val" / "good",
            self.quality_dir / "val" / "minor_defect",
            self.quality_dir / "val" / "major_defect",
            self.quality_dir / "test" / "good",
            self.quality_dir / "test" / "minor_defect",
            self.quality_dir / "test" / "major_defect"
        ]

        all_dirs = detection_dirs + quality_dirs

        for dir_path in all_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure at {self.datasets_dir}")

    def show_download_instructions(self):
        """Display comprehensive download instructions"""

        instructions = f"""
🍎 MINNEAPPLE DATASET DOWNLOAD INSTRUCTIONS
{'='*60}

The MinneApple dataset is available from multiple sources:

1. OFFICIAL SOURCE (Recommended):
   URL: {self.minneapple_info['official_url']}

   Steps:
   a) Visit the University of Minnesota Digital Conservancy
   b) Download the complete dataset (original format)
   c) Extract to a temporary directory
   d) Run this script with --organize flag to convert to YOLO format

2. ROBOFLOW (YOLO Format Ready):
   URL: {self.minneapple_info['roboflow_url']}

   Steps:
   a) Create free Roboflow account
   b) Download in YOLOv8 format
   c) Extract directly to datasets/detection/

3. GITHUB REPOSITORY:
   URL: {self.minneapple_info['github_url']}

   Contains tools and scripts for dataset manipulation

📁 TARGET DIRECTORY STRUCTURE:
{self.datasets_dir.absolute()}
├── detection/
│   ├── images/
│   │   ├── train/     # Training images
│   │   ├── val/       # Validation images
│   │   └── test/      # Test images
│   ├── labels/
│   │   ├── train/     # YOLO format labels (.txt)
│   │   ├── val/       # YOLO format labels (.txt)
│   │   └── test/      # YOLO format labels (.txt)
│   └── dataset.yaml   # Dataset configuration
└── quality/
    ├── train/
    │   ├── good/           # High quality apples
    │   ├── minor_defect/   # Minor defects
    │   └── major_defect/   # Major defects
    ├── val/
    └── test/

💡 QUICK START OPTIONS:

Option A - Use sample images for testing:
   python scripts/download_minneapple_dataset.py --create-samples

Option B - Download from Roboflow (requires account):
   python scripts/download_minneapple_dataset.py --roboflow-download

Option C - Organize existing downloaded data:
   python scripts/download_minneapple_dataset.py --organize-existing /path/to/downloaded/data

🎯 DATASET STATISTICS:
   • Total Images: {self.minneapple_info['total_images']}
   • Total Instances: {self.minneapple_info['total_instances']}
   • Classes: Apple (detection), Good/Minor/Major defect (quality)
   • Format: YOLO (detection), ImageFolder (quality classification)
"""

        print(instructions)

    def create_sample_dataset(self, num_samples: int = 50):
        """
        Create a sample dataset for testing using YOLO's sample images

        Args:
            num_samples: Number of sample images to create
        """
        logger.info(f"Creating sample dataset with {num_samples} images...")

        try:
            from ultralytics import YOLO
            from ultralytics.data.utils import download
            import random

            # Download COCO sample images
            sample_dir = self.datasets_dir / "temp_samples"
            sample_dir.mkdir(exist_ok=True)

            # Use some built-in sample images from ultralytics
            model = YOLO('yolov8n.pt')

            # Create synthetic apple dataset
            self.create_synthetic_apple_data(num_samples)

            # Create dataset.yaml
            self.create_dataset_yaml()

            logger.info(f"✅ Sample dataset created with {num_samples} images")
            logger.info(f"📁 Location: {self.detection_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to create sample dataset: {e}")

    def create_synthetic_apple_data(self, num_samples: int):
        """
        Create synthetic apple detection data for testing

        Args:
            num_samples: Number of images to create
        """
        import random

        logger.info("Creating synthetic apple detection data...")

        # Distribute samples across train/val/test
        train_count = int(num_samples * 0.8)
        val_count = int(num_samples * 0.1)
        test_count = num_samples - train_count - val_count

        splits = {
            'train': train_count,
            'val': val_count,
            'test': test_count
        }

        for split, count in splits.items():
            images_dir = self.detection_dir / "images" / split
            labels_dir = self.detection_dir / "labels" / split

            for i in range(count):
                # Create a simple synthetic image (green background with red circles as "apples")
                img = np.ones((640, 640, 3), dtype=np.uint8) * 50  # Dark background
                img[:, :, 1] = 100  # Slightly green background

                # Add random "apple" circles
                num_apples = random.randint(1, 8)
                labels = []

                for _ in range(num_apples):
                    # Random apple position and size
                    center_x = random.randint(50, 590)
                    center_y = random.randint(50, 590)
                    radius = random.randint(20, 50)

                    # Draw red circle (synthetic apple)
                    cv2.circle(img, (center_x, center_y), radius, (0, 0, 200), -1)
                    cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 2)

                    # Convert to YOLO format (normalized coordinates)
                    x_center = center_x / 640.0
                    y_center = center_y / 640.0
                    width = (radius * 2) / 640.0
                    height = (radius * 2) / 640.0

                    # YOLO format: class_id x_center y_center width height
                    labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Save image
                image_name = f"synthetic_apple_{split}_{i:03d}.jpg"
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), img)

                # Save label
                label_name = f"synthetic_apple_{split}_{i:03d}.txt"
                label_path = labels_dir / label_name

                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))

        logger.info(f"✅ Created synthetic data: {train_count} train, {val_count} val, {test_count} test images")

    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""

        yaml_content = f"""# MinneApple Dataset Configuration
path: {self.detection_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Class names
names: ['apple']

# Dataset information
download: |
  # MinneApple dataset
  # Original source: https://conservancy.umn.edu/handle/11299/206575
  # For custom data, place images in images/ and labels in labels/
  # Use YOLO format: class_id x_center y_center width height (normalized)
"""

        yaml_path = self.detection_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"✅ Created dataset.yaml: {yaml_path}")

    def check_dataset_status(self) -> Dict:
        """
        Check current dataset status

        Returns:
            Dictionary with dataset statistics
        """
        status = {
            'detection': {
                'train': {'images': 0, 'labels': 0},
                'val': {'images': 0, 'labels': 0},
                'test': {'images': 0, 'labels': 0}
            },
            'quality': {
                'train': {'good': 0, 'minor_defect': 0, 'major_defect': 0},
                'val': {'good': 0, 'minor_defect': 0, 'major_defect': 0},
                'test': {'good': 0, 'minor_defect': 0, 'major_defect': 0}
            }
        }

        # Check detection dataset
        for split in ['train', 'val', 'test']:
            images_dir = self.detection_dir / "images" / split
            labels_dir = self.detection_dir / "labels" / split

            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                status['detection'][split]['images'] = len(image_files)

            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                status['detection'][split]['labels'] = len(label_files)

        # Check quality dataset
        for split in ['train', 'val', 'test']:
            for quality in ['good', 'minor_defect', 'major_defect']:
                quality_dir = self.quality_dir / split / quality
                if quality_dir.exists():
                    image_files = list(quality_dir.glob("*.jpg")) + list(quality_dir.glob("*.png"))
                    status['quality'][split][quality] = len(image_files)

        return status

    def print_dataset_status(self):
        """Print current dataset status"""
        status = self.check_dataset_status()

        print("\n📊 DATASET STATUS")
        print("="*50)

        # Detection dataset status
        print("🎯 DETECTION DATASET:")
        for split in ['train', 'val', 'test']:
            images = status['detection'][split]['images']
            labels = status['detection'][split]['labels']
            match = "✅" if images == labels else "⚠️"
            print(f"   {split:5}: {images:4} images, {labels:4} labels {match}")

        total_images = sum(status['detection'][split]['images'] for split in ['train', 'val', 'test'])
        total_labels = sum(status['detection'][split]['labels'] for split in ['train', 'val', 'test'])

        print(f"   Total: {total_images:4} images, {total_labels:4} labels")

        # Quality dataset status
        print("\n🍎 QUALITY DATASET:")
        for split in ['train', 'val', 'test']:
            good = status['quality'][split]['good']
            minor = status['quality'][split]['minor_defect']
            major = status['quality'][split]['major_defect']
            total_split = good + minor + major
            print(f"   {split:5}: {total_split:4} total ({good} good, {minor} minor, {major} major)")

        # Dataset configuration status
        dataset_yaml = self.detection_dir / "dataset.yaml"
        config_status = "✅" if dataset_yaml.exists() else "❌"
        print(f"\n⚙️  CONFIGURATION:")
        print(f"   dataset.yaml: {config_status}")

        print("\n💡 RECOMMENDATIONS:")
        if total_images == 0:
            print("   • No images found. Run with --create-samples for testing")
            print("   • Or download real dataset following the instructions above")
        elif total_images < 100:
            print("   • Small dataset detected. Consider downloading more data")
        else:
            print("   • Dataset looks good! Ready for training and testing")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='MinneApple Dataset Downloader')

    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create synthetic sample dataset for testing'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of sample images to create (default: 50)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check current dataset status'
    )
    parser.add_argument(
        '--instructions',
        action='store_true',
        help='Show download instructions'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = MinneAppleDatasetDownloader()

    if args.instructions or not any([args.create_samples, args.status]):
        # Show instructions by default
        downloader.show_download_instructions()

    if args.status:
        downloader.print_dataset_status()

    if args.create_samples:
        downloader.create_sample_dataset(args.num_samples)
        downloader.print_dataset_status()


if __name__ == "__main__":
    main()