"""
MinneApple Model Testing Script for Presentation Results

This script tests the MinneApple YOLOv8 pre-trained model and generates
comprehensive results suitable for presentation including visualizations,
performance metrics, and comparison analysis.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.apple_detector import AppleDetector
from src.pipeline.apple_pipeline import ApplePipeline
from src.utils.evaluation import DetectionEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinneAppleModelTester:
    """
    Comprehensive testing class for MinneApple YOLOv8 model
    """

    def __init__(self, output_dir: str = "results/minneapple_test"):
        """
        Initialize the tester

        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir = self.output_dir / "metrics"

        for dir_path in [self.images_dir, self.plots_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)

        # Initialize models for comparison
        self.models = {}
        self.test_results = {}

        logger.info(f"MinneApple Model Tester initialized. Output: {self.output_dir}")

    def setup_models(self):
        """Initialize different model configurations for comparison"""

        logger.info("Setting up models for comparison...")

        # MinneApple model (primary)
        try:
            self.models['minneapple'] = AppleDetector(
                use_minneapple_weights=True,
                conf_threshold=0.3,
                device='auto'
            )
            logger.info("✅ MinneApple model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load MinneApple model: {e}")
            self.models['minneapple'] = None

        # Standard YOLOv8 for comparison
        try:
            self.models['yolov8_standard'] = AppleDetector(
                use_minneapple_weights=False,
                model_size='n',
                conf_threshold=0.3,
                device='auto'
            )
            logger.info("✅ Standard YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load standard YOLOv8: {e}")
            self.models['yolov8_standard'] = None

        # Count successfully loaded models
        loaded_models = sum(1 for model in self.models.values() if model is not None)
        logger.info(f"Successfully loaded {loaded_models}/{len(self.models)} models")

    def run_single_image_test(
        self,
        image_path: str,
        model_name: str = 'minneapple'
    ) -> Dict:
        """
        Test model on a single image and generate detailed results

        Args:
            image_path: Path to test image
            model_name: Name of model to use

        Returns:
            Dictionary with detection results and metrics
        """
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Model '{model_name}' not available")

        detector = self.models[model_name]

        logger.info(f"Testing {model_name} on {image_path}")

        # Run detection
        start_time = time.time()
        results = detector.detect(image_path)
        inference_time = time.time() - start_time

        if not results:
            logger.warning(f"No detections for {image_path}")
            return {}

        result = results[0]

        # Load and annotate image
        image = cv2.imread(image_path)
        annotated_image = detector.annotate_image(image, result)

        # Save annotated image
        image_name = Path(image_path).stem
        output_path = self.images_dir / f"{image_name}_{model_name}_result.jpg"
        cv2.imwrite(str(output_path), annotated_image)

        # Compile results
        test_result = {
            'image_path': image_path,
            'model_name': model_name,
            'model_source': getattr(detector, 'model_source', 'Unknown'),
            'total_apples': result['total_apples'],
            'confidence_scores': result['confidence_scores'],
            'average_confidence': np.mean(result['confidence_scores']) if result['confidence_scores'] else 0.0,
            'inference_time_ms': inference_time * 1000,
            'fps': 1.0 / inference_time if inference_time > 0 else 0,
            'image_shape': result['image_shape'],
            'annotated_image_path': str(output_path),
            'detections': []
        }

        # Add detailed detection info
        for i, box in enumerate(result['boxes']):
            detection = {
                'id': i,
                'bbox': [box['x1'], box['y1'], box['x2'], box['y2']],
                'center': [box['center_x'], box['center_y']],
                'area': box['area'],
                'confidence': result['confidence_scores'][i]
            }
            test_result['detections'].append(detection)

        logger.info(f"✅ Detected {result['total_apples']} apples in {inference_time*1000:.1f}ms")

        return test_result

    def run_batch_test(
        self,
        image_dir: str,
        model_name: str = 'minneapple',
        max_images: int = 10
    ) -> List[Dict]:
        """
        Test model on multiple images

        Args:
            image_dir: Directory containing test images
            model_name: Name of model to use
            max_images: Maximum number of images to test

        Returns:
            List of test results
        """
        image_dir = Path(image_dir)

        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))

        image_files = image_files[:max_images]

        logger.info(f"Running batch test on {len(image_files)} images with {model_name}")

        batch_results = []

        for image_path in image_files:
            try:
                result = self.run_single_image_test(str(image_path), model_name)
                if result:
                    batch_results.append(result)
            except Exception as e:
                logger.error(f"Error testing {image_path}: {e}")
                continue

        logger.info(f"✅ Batch test completed. {len(batch_results)}/{len(image_files)} successful")

        return batch_results

    def compare_models(self, image_path: str) -> Dict:
        """
        Compare all available models on the same image

        Args:
            image_path: Path to test image

        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }

        logger.info(f"Comparing models on {image_path}")

        for model_name, model in self.models.items():
            if model is not None:
                try:
                    result = self.run_single_image_test(image_path, model_name)
                    comparison_results['models'][model_name] = result
                    logger.info(f"✅ {model_name}: {result.get('total_apples', 0)} apples, "
                              f"{result.get('inference_time_ms', 0):.1f}ms")
                except Exception as e:
                    logger.error(f"❌ {model_name} failed: {e}")
                    comparison_results['models'][model_name] = {'error': str(e)}

        return comparison_results

    def generate_performance_plots(self, results: List[Dict], model_name: str):
        """
        Generate performance visualization plots

        Args:
            results: List of test results
            model_name: Name of the model
        """
        if not results:
            logger.warning("No results to plot")
            return

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name.title()} Model Performance Analysis', fontsize=16, fontweight='bold')

        # Extract data
        apple_counts = [r['total_apples'] for r in results]
        inference_times = [r['inference_time_ms'] for r in results]
        confidence_scores = [r['average_confidence'] for r in results]
        fps_values = [r['fps'] for r in results]

        # 1. Apple count distribution
        ax1.hist(apple_counts, bins=max(1, len(set(apple_counts))), alpha=0.7, color='green', edgecolor='black')
        ax1.set_title('Apple Count Distribution')
        ax1.set_xlabel('Number of Apples')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # 2. Inference time analysis
        ax2.plot(range(len(inference_times)), inference_times, 'b-o', markersize=4, linewidth=1)
        ax2.set_title('Inference Time per Image')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=np.mean(inference_times), color='r', linestyle='--',
                   label=f'Avg: {np.mean(inference_times):.1f}ms')
        ax2.legend()

        # 3. Confidence score distribution
        ax3.hist(confidence_scores, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Average Confidence Distribution')
        ax3.set_xlabel('Average Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. FPS performance
        ax4.bar(range(len(fps_values)), fps_values, alpha=0.7, color='purple')
        ax4.set_title('FPS Performance')
        ax4.set_xlabel('Image Index')
        ax4.set_ylabel('FPS')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=np.mean(fps_values), color='r', linestyle='--',
                   label=f'Avg: {np.mean(fps_values):.1f} FPS')
        ax4.legend()

        plt.tight_layout()

        # Save plot
        plot_path = self.plots_dir / f"{model_name}_performance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Performance plot saved: {plot_path}")

        plt.show()

    def generate_summary_report(self, results: List[Dict], model_name: str) -> Dict:
        """
        Generate comprehensive summary report

        Args:
            results: List of test results
            model_name: Name of the model

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        # Calculate summary statistics
        total_images = len(results)
        total_apples = sum(r['total_apples'] for r in results)
        avg_apples_per_image = total_apples / total_images if total_images > 0 else 0

        inference_times = [r['inference_time_ms'] for r in results]
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)

        confidence_scores = [r['average_confidence'] for r in results]
        avg_confidence = np.mean(confidence_scores)

        fps_values = [r['fps'] for r in results]
        avg_fps = np.mean(fps_values)

        # Compile summary
        summary = {
            'model_name': model_name,
            'model_source': results[0].get('model_source', 'Unknown'),
            'test_summary': {
                'total_images_tested': total_images,
                'total_apples_detected': total_apples,
                'avg_apples_per_image': round(avg_apples_per_image, 2),
                'detection_rate': f"{(sum(1 for r in results if r['total_apples'] > 0) / total_images * 100):.1f}%"
            },
            'performance_metrics': {
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'std_inference_time_ms': round(std_inference_time, 2),
                'min_inference_time_ms': round(min(inference_times), 2),
                'max_inference_time_ms': round(max(inference_times), 2),
                'avg_fps': round(avg_fps, 2),
                'target_fps_achieved': avg_fps >= 15  # Based on your config target
            },
            'detection_quality': {
                'avg_confidence_score': round(avg_confidence, 3),
                'min_confidence': round(min(confidence_scores), 3),
                'max_confidence': round(max(confidence_scores), 3),
                'high_confidence_detections': f"{(sum(1 for c in confidence_scores if c > 0.7) / len(confidence_scores) * 100):.1f}%"
            },
            'timestamp': datetime.now().isoformat()
        }

        # Save summary
        summary_path = self.metrics_dir / f"{model_name}_summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"📋 Summary report saved: {summary_path}")

        return summary

    def print_presentation_summary(self, summary: Dict):
        """
        Print a formatted summary suitable for presentation

        Args:
            summary: Summary dictionary from generate_summary_report
        """
        print("\n" + "="*80)
        print(f"🍎 MINNEAPPLE YOLOV8 MODEL TEST RESULTS")
        print("="*80)
        print(f"Model: {summary['model_name'].title()}")
        print(f"Source: {summary['model_source']}")
        print(f"Test Date: {summary['timestamp'][:19]}")
        print("\n📊 DETECTION SUMMARY:")
        print(f"  • Total Images Tested: {summary['test_summary']['total_images_tested']}")
        print(f"  • Total Apples Detected: {summary['test_summary']['total_apples_detected']}")
        print(f"  • Average Apples per Image: {summary['test_summary']['avg_apples_per_image']}")
        print(f"  • Detection Success Rate: {summary['test_summary']['detection_rate']}")

        print("\n⚡ PERFORMANCE METRICS:")
        perf = summary['performance_metrics']
        print(f"  • Average Inference Time: {perf['avg_inference_time_ms']:.1f} ms")
        print(f"  • Average FPS: {perf['avg_fps']:.1f}")
        print(f"  • Target FPS (15+) Achieved: {'✅ Yes' if perf['target_fps_achieved'] else '❌ No'}")
        print(f"  • Time Range: {perf['min_inference_time_ms']:.1f} - {perf['max_inference_time_ms']:.1f} ms")

        print("\n🎯 DETECTION QUALITY:")
        qual = summary['detection_quality']
        print(f"  • Average Confidence: {qual['avg_confidence_score']:.3f}")
        print(f"  • Confidence Range: {qual['min_confidence']:.3f} - {qual['max_confidence']:.3f}")
        print(f"  • High Confidence (>0.7): {qual['high_confidence_detections']}")

        print("\n💡 KEY INSIGHTS:")
        if perf['avg_fps'] >= 15:
            print(f"  ✅ Model achieves real-time performance ({perf['avg_fps']:.1f} FPS > 15 FPS)")
        else:
            print(f"  ⚠️  Model below real-time threshold ({perf['avg_fps']:.1f} FPS < 15 FPS)")

        if qual['avg_confidence_score'] > 0.7:
            print(f"  ✅ High detection confidence ({qual['avg_confidence_score']:.3f} > 0.7)")
        else:
            print(f"  ⚠️  Detection confidence could be improved ({qual['avg_confidence_score']:.3f} < 0.7)")

        print("="*80)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test MinneApple YOLOv8 Model')

    parser.add_argument(
        '--test-dir',
        type=str,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--single-image',
        type=str,
        help='Single image to test'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=10,
        help='Maximum number of images to test in batch mode'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/minneapple_test',
        help='Output directory for results'
    )
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare MinneApple vs standard YOLOv8'
    )

    args = parser.parse_args()

    # Initialize tester
    tester = MinneAppleModelTester(args.output_dir)
    tester.setup_models()

    # Run tests based on arguments
    if args.single_image:
        logger.info(f"Testing single image: {args.single_image}")

        if args.compare_models:
            # Compare models
            comparison = tester.compare_models(args.single_image)
            print("\n🔄 MODEL COMPARISON RESULTS:")
            for model_name, result in comparison['models'].items():
                if 'error' not in result:
                    print(f"{model_name}: {result['total_apples']} apples, {result['inference_time_ms']:.1f}ms")
                else:
                    print(f"{model_name}: Error - {result['error']}")
        else:
            # Single model test
            result = tester.run_single_image_test(args.single_image, 'minneapple')
            summary = tester.generate_summary_report([result], 'minneapple')
            tester.print_presentation_summary(summary)

    elif args.test_dir:
        logger.info(f"Testing directory: {args.test_dir}")

        # Batch test
        results = tester.run_batch_test(args.test_dir, 'minneapple', args.max_images)

        if results:
            # Generate comprehensive results
            summary = tester.generate_summary_report(results, 'minneapple')
            tester.generate_performance_plots(results, 'minneapple')
            tester.print_presentation_summary(summary)
        else:
            logger.error("No successful test results to analyze")

    else:
        logger.error("Please specify either --single-image or --test-dir")
        parser.print_help()


if __name__ == "__main__":
    main()