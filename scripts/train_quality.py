"""
Training Script for Apple Quality Classification Model using MobileNetV3

This script provides comprehensive training functionality for the apple quality 
classification model with data loading, augmentation, and evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.quality_classifier import QualityClassificationPipeline, QualityClassifier
from src.data.data_processing import QualityDataset, get_quality_transforms

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


def create_data_loaders(config: dict):
    """Create data loaders for training, validation, and testing"""
    data_config = config.get('data', {}).get('quality_classification', {})
    training_config = config.get('training', {}).get('quality_classification', {})
    
    # Get dataset paths
    dataset_root = data_config.get('dataset_path', 'datasets/quality')
    
    # Create transforms
    train_transform = get_quality_transforms(augment=True, image_size=224)
    val_transform = get_quality_transforms(augment=False, image_size=224)
    
    # Create datasets
    train_dataset = QualityDataset(
        root_dir=os.path.join(dataset_root, 'train'),
        transform=train_transform,
        class_names=data_config.get('classes', ['good', 'minor_defect', 'major_defect'])
    )
    
    val_dataset = QualityDataset(
        root_dir=os.path.join(dataset_root, 'val'),
        transform=val_transform,
        class_names=data_config.get('classes', ['good', 'minor_defect', 'major_defect'])
    )
    
    test_dataset = QualityDataset(
        root_dir=os.path.join(dataset_root, 'test'),
        transform=val_transform,
        class_names=data_config.get('classes', ['good', 'minor_defect', 'major_defect'])
    )
    
    # Create data loaders
    batch_size = training_config.get('batch_size', 32)
    num_workers = training_config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def train_quality_model(config_path: str, **kwargs):
    """
    Train apple quality classification model
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional training parameters
    """
    # Load configuration
    config = load_config(config_path)
    training_config = config.get('training', {}).get('quality_classification', {})
    
    # Override config with command line arguments
    for key, value in kwargs.items():
        if value is not None:
            training_config[key] = value
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Initialize pipeline
    pipeline = QualityClassificationPipeline(
        device=training_config.get('device', 'auto')
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/quality/train_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training configuration
    config_save_path = os.path.join(output_dir, 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Starting quality classification model training...")
    logger.info(f"Training configuration: {training_config}")
    
    try:
        # Start training
        training_history = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config.get('epochs', 50),
            learning_rate=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-4),
            save_best=True,
            save_path=os.path.join(output_dir, 'best_model.pth')
        )
        
        logger.info("Training completed successfully!")
        
        # Plot training history
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        pipeline.plot_training_history(save_path=history_plot_path)
        
        # Evaluate on test set
        logger.info("Starting model evaluation on test set...")
        evaluation_results = pipeline.evaluate(test_loader)
        
        logger.info("Evaluation results:")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Macro F1: {evaluation_results['macro_f1']:.4f}")
        logger.info(f"  Macro Precision: {evaluation_results['macro_precision']:.4f}")
        logger.info(f"  Macro Recall: {evaluation_results['macro_recall']:.4f}")
        
        # Per-class metrics
        logger.info("Per-class metrics:")
        for class_name, metrics in evaluation_results['per_class_metrics'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1: {metrics['f1']:.4f}")
            logger.info(f"    Support: {metrics['support']}")
        
        # Plot confusion matrix
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        pipeline.plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            save_path=confusion_matrix_path
        )
        
        # Save evaluation results
        eval_results_path = os.path.join(output_dir, 'evaluation_results.yaml')
        eval_results_clean = {}
        for key, value in evaluation_results.items():
            if key != 'confusion_matrix':  # Skip numpy array for YAML
                eval_results_clean[key] = value
        
        with open(eval_results_path, 'w') as f:
            yaml.dump(eval_results_clean, f, default_flow_style=False)
        
        # Save final model
        final_model_path = os.path.join(output_dir, 'final_model.pth')
        pipeline.save_model(final_model_path)
        
        # Save model info
        model_info = {
            'model_size_mb': pipeline.model.get_model_size(),
            'num_parameters': sum(p.numel() for p in pipeline.model.parameters()),
            'num_classes': len(pipeline.model.class_names),
            'class_names': pipeline.model.class_names
        }
        
        model_info_path = os.path.join(output_dir, 'model_info.yaml')
        with open(model_info_path, 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
        
        logger.info(f"Training artifacts saved to {output_dir}")
        logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        return {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'model_info': model_info,
            'output_directory': output_dir,
            'model_path': final_model_path
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Apple Quality Classification Model')
    
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
        '--weight-decay', 
        type=float,
        help='Weight decay for regularization'
    )
    parser.add_argument(
        '--device', 
        type=str,
        help='Training device (auto, cpu, cuda, mps)'
    )
    
    args = parser.parse_args()
    
    # Convert args to dict for passing to training function
    kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device
    }
    
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    try:
        results = train_quality_model(args.config, **kwargs)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Output directory: {results['output_directory']}")
        print(f"Model saved to: {results['model_path']}")
        print(f"Model size: {results['model_info']['model_size_mb']:.2f} MB")
        print("\nEvaluation metrics:")
        eval_results = results['evaluation_results']
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  Macro F1: {eval_results['macro_f1']:.4f}")
        print(f"  Macro Precision: {eval_results['macro_precision']:.4f}")
        print(f"  Macro Recall: {eval_results['macro_recall']:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()