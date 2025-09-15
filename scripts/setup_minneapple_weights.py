"""
MinneApple Weights Setup Script

This script helps download and setup the MinneApple YOLOv8 pre-trained weights
for immediate use in the apple detection pipeline.
"""

import os
import sys
import requests
import logging
import shutil
from pathlib import Path
from typing import List, Dict
import zipfile
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinneAppleWeightsSetup:
    """
    Setup class for MinneApple YOLOv8 weights
    """

    def __init__(self, project_root: str = None):
        """
        Initialize the setup

        Args:
            project_root: Root directory of the project
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent

        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"
        self.minneapple_dir = self.models_dir / "minneapple"

        # Create directories
        self.minneapple_dir.mkdir(parents=True, exist_ok=True)

        # MinneApple repository information
        self.repo_url = "https://github.com/joy0010/Apple-Detection-in-MinneApple-Dataset-with-YOLOv8"
        self.available_models = {
            "Original_original_best.pt": {
                "description": "Original dataset with original training",
                "size": "~50MB",
                "performance": "Standard accuracy"
            },
            "Original_augmented_best.pt": {
                "description": "Original dataset with data augmentation",
                "size": "~50MB",
                "performance": "Enhanced accuracy (recommended)"
            },
            "Small_original_best.pt": {
                "description": "Small subset, original training",
                "size": "~50MB",
                "performance": "Faster inference, lower accuracy"
            },
            "Small_augmented_best.pt": {
                "description": "Small subset with augmentation",
                "size": "~50MB",
                "performance": "Balanced speed/accuracy"
            },
            "Ground_original_best.pt": {
                "description": "Fallen apple detection, original training",
                "size": "~50MB",
                "performance": "Specialized for ground apples"
            },
            "Ground_augmented_best.pt": {
                "description": "Fallen apple detection with augmentation",
                "size": "~50MB",
                "performance": "Best for ground apple detection"
            }
        }

        logger.info(f"MinneApple weights setup initialized")
        logger.info(f"Target directory: {self.minneapple_dir}")

    def check_existing_weights(self) -> List[str]:
        """
        Check which weights are already downloaded

        Returns:
            List of existing weight files
        """
        existing_weights = []

        for model_name in self.available_models.keys():
            model_path = self.minneapple_dir / model_name
            if model_path.exists():
                existing_weights.append(model_name)

        if existing_weights:
            logger.info(f"Found existing weights: {existing_weights}")
        else:
            logger.info("No existing MinneApple weights found")

        return existing_weights

    def display_available_models(self):
        """Display information about available models"""
        print("\n🍎 MINNEAPPLE YOLOV8 MODELS")
        print("="*60)
        print("Available pre-trained models:")
        print()

        for i, (model_name, info) in enumerate(self.available_models.items(), 1):
            status = "✅ Downloaded" if (self.minneapple_dir / model_name).exists() else "❌ Not downloaded"
            print(f"{i}. {model_name}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Performance: {info['performance']}")
            print(f"   Status: {status}")
            print()

    def create_download_instructions(self) -> str:
        """
        Create instructions for manual download

        Returns:
            Instructions text
        """
        instructions = f"""
🔧 MINNEAPPLE WEIGHTS SETUP INSTRUCTIONS

Since direct download is not available, please follow these steps:

1. Visit the MinneApple repository:
   {self.repo_url}

2. Navigate to the "Models" directory in the repository

3. Download the model files you need:
   Recommended: Original_augmented_best.pt (best overall performance)

4. Place the downloaded .pt files in:
   {self.minneapple_dir.absolute()}

5. Verify installation by running:
   python scripts/validate_setup.py

Available Models:
"""

        for model_name, info in self.available_models.items():
            status = "✅" if (self.minneapple_dir / model_name).exists() else "❌"
            instructions += f"   {status} {model_name} - {info['description']}\n"

        instructions += f"""
💡 QUICK TEST:
After downloading, test the model with:
   python scripts/test_minneapple_model.py --single-image path/to/test/image.jpg

📧 If you need help, check the repository README or issues section.
"""

        return instructions

    def validate_downloaded_weights(self) -> Dict[str, bool]:
        """
        Validate downloaded weight files

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        for model_name in self.available_models.keys():
            model_path = self.minneapple_dir / model_name

            if not model_path.exists():
                validation_results[model_name] = False
                continue

            # Basic validation - check file size
            file_size = model_path.stat().st_size
            if file_size < 1024 * 1024:  # Less than 1MB probably corrupted
                logger.warning(f"{model_name} seems corrupted (size: {file_size} bytes)")
                validation_results[model_name] = False
            else:
                validation_results[model_name] = True

        return validation_results

    def test_model_loading(self, model_name: str = "Original_augmented_best.pt") -> bool:
        """
        Test if a model can be loaded successfully

        Args:
            model_name: Name of the model to test

        Returns:
            True if model loads successfully
        """
        model_path = self.minneapple_dir / model_name

        if not model_path.exists():
            logger.error(f"Model {model_name} not found at {model_path}")
            return False

        try:
            # Add src to path for imports
            sys.path.append(str(self.project_root))
            from src.models.apple_detector import AppleDetector

            # Try to load the model
            logger.info(f"Testing model loading: {model_name}")
            detector = AppleDetector(model_path=str(model_path), use_minneapple_weights=False)

            logger.info(f"✅ Successfully loaded {model_name}")
            logger.info(f"Model source: {getattr(detector, 'model_source', 'Unknown')}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return False

    def create_symlinks(self):
        """Create convenient symlinks for commonly used models"""
        # Create symlink to the recommended model
        recommended_model = "Original_augmented_best.pt"
        recommended_path = self.minneapple_dir / recommended_model

        if recommended_path.exists():
            symlink_path = self.models_dir / "minneapple_best.pt"

            try:
                if symlink_path.exists() or symlink_path.is_symlink():
                    symlink_path.unlink()

                # Create relative symlink
                relative_target = Path("minneapple") / recommended_model
                symlink_path.symlink_to(relative_target)

                logger.info(f"✅ Created symlink: {symlink_path} -> {relative_target}")

            except Exception as e:
                logger.warning(f"Could not create symlink: {e}")

    def run_setup(self):
        """Run the complete setup process"""
        print("🍎 MinneApple YOLOv8 Weights Setup")
        print("="*50)

        # Check existing weights
        existing = self.check_existing_weights()

        # Display available models
        self.display_available_models()

        # Validate any existing weights
        if existing:
            print("\n🔍 Validating existing weights...")
            validation = self.validate_downloaded_weights()

            for model_name, is_valid in validation.items():
                if model_name in existing:
                    status = "✅ Valid" if is_valid else "❌ Invalid/Corrupted"
                    print(f"   {model_name}: {status}")

        # Show download instructions
        print(self.create_download_instructions())

        # Test model loading if we have weights
        if existing:
            print("🧪 Testing model loading...")
            test_model = existing[0]  # Test first available model
            if self.test_model_loading(test_model):
                print(f"✅ Model loading test passed with {test_model}")

                # Create convenience symlinks
                self.create_symlinks()
            else:
                print(f"❌ Model loading test failed with {test_model}")

        print("\n🎉 Setup process completed!")

        if not existing:
            print("\n⚠️  No models found. Please download weights following the instructions above.")
        else:
            print(f"\n✅ Ready to use! Found {len(existing)} model(s).")
            print("\n🚀 Quick start:")
            print("   python scripts/test_minneapple_model.py --single-image your_image.jpg")


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description='Setup MinneApple YOLOv8 weights')
    parser.add_argument(
        '--project-root',
        type=str,
        help='Project root directory (default: auto-detect)'
    )
    parser.add_argument(
        '--test-loading',
        type=str,
        help='Test loading a specific model file'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing weights'
    )

    args = parser.parse_args()

    # Initialize setup
    setup = MinneAppleWeightsSetup(args.project_root)

    if args.validate_only:
        # Only validate existing weights
        validation = setup.validate_downloaded_weights()
        for model_name, is_valid in validation.items():
            status = "✅ Valid" if is_valid else "❌ Invalid"
            if (setup.minneapple_dir / model_name).exists():
                print(f"{model_name}: {status}")

    elif args.test_loading:
        # Test specific model
        success = setup.test_model_loading(args.test_loading)
        sys.exit(0 if success else 1)

    else:
        # Run full setup
        setup.run_setup()


if __name__ == "__main__":
    main()