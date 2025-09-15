"""
Training Script for Apple Detection Model using YOLOv8

This script provides comprehensive training functionality for the apple detection model
with configuration management, logging, and evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.apple_detector import AppleDetector
from src.data.data_processing import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_dataset(config: dict, data_manager: DataManager) -> str:
    """Setup and prepare dataset for training"""
    dataset_config = config.get('data', {}).get('detection', {})
    
    # Create dataset YAML file
    dataset_yaml_path = 'dataset_detection.yaml'
    
    dataset_yaml_content = f"""
path: {dataset_config.get('dataset_path', 'datasets/detection')}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['apple']
"""
    
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    
    logger.info(f"Dataset configuration saved to {dataset_yaml_path}")
    return dataset_yaml_path


def train_detection_model(config_path: str, **kwargs):
    """
    Train apple detection model
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional training parameters
    """
    # Load configuration
    config = load_config(config_path)
    training_config = config.get('training', {}).get('detection', {})
    
    # Override config with command line arguments
    for key, value in kwargs.items():
        if value is not None:
            training_config[key] = value
    
    # Setup data manager
    data_manager = DataManager(config_path)
    
    # Setup dataset
    dataset_yaml = setup_dataset(config, data_manager)
    
    # Initialize detector
    detector = AppleDetector(
        model_size=training_config.get('model_size', 'n'),
        device=training_config.get('device', 'auto')
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/detect/train_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training configuration
    config_save_path = os.path.join(output_dir, 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Starting detection model training...")
    logger.info(f"Training configuration: {training_config}")
    
    # Start training
    try:
        results = detector.train(
            data_yaml=dataset_yaml,
            epochs=training_config.get('epochs', 100),
            batch_size=training_config.get('batch_size', 16),
            learning_rate=training_config.get('learning_rate', 0.001),
            save_dir=output_dir,
            patience=training_config.get('patience', 20),
            save_period=training_config.get('save_period', 10)
        )
        
        logger.info("Training completed successfully!")
        
        # Evaluate model
        logger.info("Starting model evaluation...")
        metrics = detector.validate(dataset_yaml)
        
        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save evaluation results
        eval_results_path = os.path.join(output_dir, 'evaluation_results.yaml')
        with open(eval_results_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.pt')
        detector.save_model(final_model_path)
        
        logger.info(f"Training artifacts saved to {output_dir}")
        
        return {
            'training_results': results,
            'evaluation_metrics': metrics,
            'output_directory': output_dir,
            'model_path': final_model_path
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Apple Detection Model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', 
        type=int,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--device', 
        type=str,
        help='Training device (auto, cpu, cuda, mps)'
    )
    parser.add_argument(
        '--model-size', 
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model size'
    )
    
    args = parser.parse_args()
    
    # Convert args to dict for passing to training function
    kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'model_size': args.model_size
    }
    
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    try:
        results = train_detection_model(args.config, **kwargs)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Output directory: {results['output_directory']}")
        print(f"Model saved to: {results['model_path']}")
        print("\nEvaluation metrics:")
        for metric, value in results['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()