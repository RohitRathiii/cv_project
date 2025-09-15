"""
Complete training script that trains both detection and quality models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train_detection import train_detection_model
from train_quality import train_quality_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_complete_pipeline(config_path: str, skip_detection: bool = False, skip_quality: bool = False):
    """
    Train complete apple detection and quality grading pipeline
    
    Args:
        config_path: Path to configuration file
        skip_detection: Skip detection model training
        skip_quality: Skip quality model training
    """
    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = f"runs/complete_pipeline_{timestamp}"
    os.makedirs(main_output_dir, exist_ok=True)
    
    results = {
        'timestamp': timestamp,
        'output_directory': main_output_dir,
        'detection_results': None,
        'quality_results': None
    }
    
    try:
        # Train detection model
        if not skip_detection:
            logger.info("="*60)
            logger.info("STARTING DETECTION MODEL TRAINING")
            logger.info("="*60)
            
            detection_results = train_detection_model(config_path)
            results['detection_results'] = detection_results
            
            logger.info("Detection model training completed successfully!")
        
        # Train quality classification model
        if not skip_quality:
            logger.info("="*60)
            logger.info("STARTING QUALITY CLASSIFICATION MODEL TRAINING")
            logger.info("="*60)
            
            quality_results = train_quality_model(config_path)
            results['quality_results'] = quality_results
            
            logger.info("Quality classification model training completed successfully!")
        
        # Save combined results
        results_path = os.path.join(main_output_dir, 'pipeline_training_results.yaml')
        with open(results_path, 'w') as f:
            # Prepare results for YAML (remove complex objects)
            yaml_results = {
                'timestamp': results['timestamp'],
                'output_directory': results['output_directory']
            }
            
            if results['detection_results']:
                yaml_results['detection'] = {
                    'output_directory': results['detection_results']['output_directory'],
                    'model_path': results['detection_results']['model_path'],
                    'evaluation_metrics': results['detection_results']['evaluation_metrics']
                }
            
            if results['quality_results']:
                yaml_results['quality'] = {
                    'output_directory': results['quality_results']['output_directory'],
                    'model_path': results['quality_results']['model_path'],
                    'model_info': results['quality_results']['model_info'],
                    'evaluation_metrics': {
                        k: v for k, v in results['quality_results']['evaluation_results'].items()
                        if k != 'confusion_matrix'
                    }
                }
            
            yaml.dump(yaml_results, f, default_flow_style=False)
        
        logger.info(f"Complete pipeline training results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline training failed: {str(e)}")
        raise


def main():
    """Main function for complete pipeline training"""
    parser = argparse.ArgumentParser(description='Train Complete Apple Detection Pipeline')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-detection',
        action='store_true',
        help='Skip detection model training'
    )
    parser.add_argument(
        '--skip-quality',
        action='store_true',
        help='Skip quality model training'
    )
    
    args = parser.parse_args()
    
    if args.skip_detection and args.skip_quality:
        print("Error: Cannot skip both detection and quality training!")
        sys.exit(1)
    
    try:
        results = train_complete_pipeline(
            config_path=args.config,
            skip_detection=args.skip_detection,
            skip_quality=args.skip_quality
        )
        
        print("\n" + "="*60)
        print("COMPLETE PIPELINE TRAINING FINISHED!")
        print("="*60)
        print(f"Main output directory: {results['output_directory']}")
        
        if results['detection_results']:
            print(f"\nDetection Model:")
            print(f"  Output: {results['detection_results']['output_directory']}")
            print(f"  Model: {results['detection_results']['model_path']}")
            metrics = results['detection_results']['evaluation_metrics']
            print(f"  mAP50: {metrics['mAP50']:.4f}")
            print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
        
        if results['quality_results']:
            print(f"\nQuality Classification Model:")
            print(f"  Output: {results['quality_results']['output_directory']}")
            print(f"  Model: {results['quality_results']['model_path']}")
            print(f"  Size: {results['quality_results']['model_info']['model_size_mb']:.2f} MB")
            metrics = results['quality_results']['evaluation_results']
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nPipeline training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()