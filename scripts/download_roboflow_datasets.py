"""
Roboflow Apple Datasets Downloader

This script helps download apple detection datasets from Roboflow Universe
in YOLO format for immediate use.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoboflowDatasetDownloader:
    """
    Download apple detection datasets from Roboflow
    """

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent

        self.project_root = Path(project_root)
        self.datasets_dir = self.project_root / "datasets" / "detection"

        # Available datasets
        self.available_datasets = {
            "lakshantha_apple": {
                "name": "Apple Detection by Lakshantha Dissanayake",
                "images": 699,
                "classes": ["apple"],
                "workspace": "lakshantha-dissanayake",
                "project": "apple-detection-5z37o",
                "version": 1,
                "description": "699 open source apple images with annotations",
                "url": "https://universe.roboflow.com/lakshantha-dissanayake/apple-detection-5z37o/dataset/1"
            },
            "psu_apple": {
                "name": "Apple Detection YOLO by Penn State",
                "images": 211,
                "classes": ["apple", "branches", "stem"],
                "workspace": "the-pennsylvania-state-university",
                "project": "apple-detection-yolo",
                "version": 1,
                "description": "211 apple detection images with branches and stems",
                "url": "https://universe.roboflow.com/the-pennsylvania-state-university/apple-detection-yolo"
            },
            "fruits_yolo": {
                "name": "Fruits by YOLO (Multi-fruit)",
                "images": 1176,
                "classes": ["apple", "banana", "grapes", "kiwi", "mango", "orange", "pineapple"],
                "workspace": "fruitsdetection",
                "project": "fruits-by-yolo",
                "version": 1,
                "description": "1176 images with 7 fruit classes including apple",
                "url": "https://universe.roboflow.com/fruitsdetection/fruits-by-yolo"
            }
        }

    def show_available_datasets(self):
        """Display available datasets"""
        print("\n🍎 AVAILABLE APPLE DETECTION DATASETS FROM ROBOFLOW")
        print("="*70)

        for key, dataset in self.available_datasets.items():
            print(f"\n📦 {dataset['name']}")
            print(f"   Images: {dataset['images']}")
            print(f"   Classes: {', '.join(dataset['classes'])}")
            print(f"   Description: {dataset['description']}")
            print(f"   URL: {dataset['url']}")
            print(f"   Download ID: {key}")

    def install_roboflow(self):
        """Install roboflow package if needed"""
        try:
            import roboflow
            logger.info("✅ Roboflow package already installed")
            return True
        except ImportError:
            logger.info("📦 Installing roboflow package...")
            try:
                os.system("pip install roboflow")
                import roboflow
                logger.info("✅ Roboflow package installed successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to install roboflow: {e}")
                print("\n❌ Please install roboflow manually:")
                print("   pip install roboflow")
                return False

    def download_dataset(self, dataset_key: str, api_key: str = None):
        """
        Download a dataset from Roboflow

        Args:
            dataset_key: Key of the dataset to download
            api_key: Roboflow API key (optional for public datasets)
        """
        if dataset_key not in self.available_datasets:
            logger.error(f"Unknown dataset: {dataset_key}")
            logger.info(f"Available datasets: {list(self.available_datasets.keys())}")
            return False

        if not self.install_roboflow():
            return False

        try:
            from roboflow import Roboflow

            dataset_info = self.available_datasets[dataset_key]
            logger.info(f"📥 Downloading {dataset_info['name']}...")

            # Initialize Roboflow
            rf = Roboflow(api_key=api_key) if api_key else Roboflow()

            # Get project
            project = rf.workspace(dataset_info['workspace']).project(dataset_info['project'])

            # Download dataset
            dataset = project.version(dataset_info['version']).download(
                "yolov8",
                location=str(self.datasets_dir),
                overwrite=True
            )

            logger.info(f"✅ Successfully downloaded {dataset_info['name']}")
            logger.info(f"📁 Location: {dataset.location}")

            # Update dataset.yaml path
            self.update_dataset_yaml(dataset.location)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_key}: {e}")
            logger.info("\n💡 Troubleshooting tips:")
            logger.info("1. Check your internet connection")
            logger.info("2. For private datasets, provide API key")
            logger.info("3. Visit the dataset URL manually to check availability")
            return False

    def update_dataset_yaml(self, dataset_location: str):
        """
        Update dataset.yaml with correct paths

        Args:
            dataset_location: Path where dataset was downloaded
        """
        try:
            yaml_path = Path(dataset_location) / "data.yaml"
            if yaml_path.exists():
                # Read existing yaml
                with open(yaml_path, 'r') as f:
                    content = f.read()

                # Update path to absolute path
                updated_content = content.replace(
                    f"path: {dataset_location}",
                    f"path: {Path(dataset_location).absolute()}"
                )

                with open(yaml_path, 'w') as f:
                    f.write(updated_content)

                # Also copy to standard location
                standard_yaml = self.datasets_dir / "dataset.yaml"
                with open(standard_yaml, 'w') as f:
                    f.write(updated_content)

                logger.info(f"✅ Updated dataset configuration: {standard_yaml}")

        except Exception as e:
            logger.warning(f"⚠️ Could not update dataset.yaml: {e}")

    def show_download_instructions(self):
        """Show step-by-step download instructions"""
        instructions = """
🚀 ROBOFLOW DATASET DOWNLOAD INSTRUCTIONS
{'='*50}

STEP 1: Choose a dataset from the list above
   • lakshantha_apple: Best for pure apple detection (699 images)
   • psu_apple: Includes branches/stems (211 images)
   • fruits_yolo: Multi-fruit including apples (1176 images)

STEP 2: Download using this script
   python scripts/download_roboflow_datasets.py --download <dataset_id>

   Examples:
   python scripts/download_roboflow_datasets.py --download lakshantha_apple
   python scripts/download_roboflow_datasets.py --download fruits_yolo

STEP 3: Test your dataset
   python scripts/test_minneapple_model.py --test-dir datasets/detection/images/test

STEP 4: Train your model (if needed)
   python scripts/train_detection.py --config config/config.yaml

💡 TIPS:
• No API key needed for public datasets
• Dataset will be downloaded in YOLOv8 format
• Images and labels automatically organized
• Ready to use immediately after download

🔧 TROUBLESHOOTING:
If download fails:
1. Install roboflow: pip install roboflow
2. Check internet connection
3. Try a different dataset
4. Visit the dataset URL manually
"""
        print(instructions)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Download Roboflow Apple Datasets')

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    parser.add_argument(
        '--download',
        type=str,
        help='Download specific dataset by ID'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Roboflow API key (optional for public datasets)'
    )
    parser.add_argument(
        '--instructions',
        action='store_true',
        help='Show download instructions'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = RoboflowDatasetDownloader()

    if args.list or not any([args.download, args.instructions]):
        downloader.show_available_datasets()

    if args.instructions:
        downloader.show_download_instructions()

    if args.download:
        success = downloader.download_dataset(args.download, args.api_key)
        if success:
            print(f"\n🎉 Dataset '{args.download}' downloaded successfully!")
            print("Ready to test with:")
            print("python scripts/test_minneapple_model.py --test-dir datasets/detection/images/test")
        else:
            print(f"\n❌ Failed to download dataset '{args.download}'")


if __name__ == "__main__":
    main()