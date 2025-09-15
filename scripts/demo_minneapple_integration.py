"""
MinneApple Integration Demo

This script demonstrates how to use the integrated MinneApple YOLOv8 weights
in your apple detection pipeline for presentation and testing.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.apple_detector import AppleDetector
from src.pipeline.apple_pipeline import ApplePipeline
from scripts.test_minneapple_model import MinneAppleModelTester
from scripts.generate_presentation_results import PresentationResultsGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_single_image(image_path: str, output_dir: str = "results/demo"):
    """
    Demo the pipeline on a single image

    Args:
        image_path: Path to test image
        output_dir: Output directory for results
    """
    print("🍎 SINGLE IMAGE DEMO")
    print("="*50)

    try:
        # Initialize detector with MinneApple weights
        detector = AppleDetector(
            use_minneapple_weights=True,
            conf_threshold=0.3,
            device='auto'
        )

        print(f"✅ Model loaded: {getattr(detector, 'model_source', 'Unknown')}")
        print(f"📍 Processing: {image_path}")

        # Run detection
        results = detector.detect(image_path)

        if results:
            result = results[0]
            print(f"🍎 Detected: {result['total_apples']} apples")

            if result['confidence_scores']:
                avg_conf = sum(result['confidence_scores']) / len(result['confidence_scores'])
                print(f"🎯 Average confidence: {avg_conf:.3f}")

            # Create annotated image
            import cv2
            image = cv2.imread(image_path)
            annotated = detector.annotate_image(image, result)

            # Save result
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            result_image_path = output_path / f"demo_result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(result_image_path), annotated)

            print(f"💾 Saved result: {result_image_path}")

            # Performance stats
            perf = detector.get_performance_stats()
            if perf:
                print(f"⚡ Performance: {perf['avg_inference_time']*1000:.1f}ms, {perf['fps']:.1f} FPS")

        else:
            print("❌ No apples detected")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_batch_processing(images_dir: str, output_dir: str = "results/demo_batch"):
    """
    Demo batch processing with comprehensive results

    Args:
        images_dir: Directory containing test images
        output_dir: Output directory for results
    """
    print("\n📁 BATCH PROCESSING DEMO")
    print("="*50)

    try:
        # Initialize tester
        tester = MinneAppleModelTester(output_dir)
        tester.setup_models()

        # Run batch test
        results = tester.run_batch_test(images_dir, model_name='minneapple', max_images=5)

        if results:
            # Generate summary
            summary = tester.generate_summary_report(results, 'minneapple')

            # Generate visualizations
            tester.generate_performance_plots(results, 'minneapple')

            # Print summary
            tester.print_presentation_summary(summary)

            # Save results for presentation generator
            results_data = {
                'model_name': 'minneapple',
                'test_results': results,
                'summary': summary,
                'timestamp': results[0].get('timestamp') if results else None
            }

            results_file = Path(output_dir) / "batch_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            print(f"\n💾 Batch results saved: {results_file}")

        else:
            print("❌ No successful batch results")

    except Exception as e:
        print(f"❌ Error in batch processing: {e}")


def demo_presentation_materials(results_file: str, output_dir: str = "results/presentation_demo"):
    """
    Demo presentation materials generation

    Args:
        results_file: Path to results JSON file
        output_dir: Output directory for presentation materials
    """
    print("\n🎨 PRESENTATION MATERIALS DEMO")
    print("="*50)

    try:
        # Load results
        with open(results_file, 'r') as f:
            results_data = json.load(f)

        # Initialize generator
        generator = PresentationResultsGenerator(output_dir)

        print("📊 Generating charts and visualizations...")

        # Generate performance dashboard
        if 'test_results' in results_data:
            dashboard_path = generator.create_performance_dashboard(
                results_data['test_results'],
                results_data.get('model_name', 'minneapple')
            )
            print(f"✅ Performance dashboard: {dashboard_path}")

            # Generate detection showcase
            showcase_path = generator.create_detection_showcase(results_data['test_results'])
            print(f"✅ Detection showcase: {showcase_path}")

        # Generate metrics table
        if 'summary' in results_data:
            table_path = generator.generate_metrics_table(results_data['summary'])
            print(f"✅ Metrics table: {table_path}")

        # Create presentation summary
        summary_path = generator.create_presentation_summary(results_data)
        print(f"✅ Presentation summary: {summary_path}")

        print(f"\n🎉 All materials ready in: {output_dir}")
        print("\n📋 Files ready for PowerPoint:")
        print("   - Performance dashboard (PNG)")
        print("   - Detection results showcase (PNG)")
        print("   - Metrics table (PNG)")
        print("   - Presentation summary (Markdown)")

    except Exception as e:
        print(f"❌ Error generating presentation materials: {e}")


def demo_model_comparison(image_path: str, output_dir: str = "results/comparison_demo"):
    """
    Demo model comparison between MinneApple and standard YOLOv8

    Args:
        image_path: Path to test image
        output_dir: Output directory for comparison results
    """
    print("\n🔄 MODEL COMPARISON DEMO")
    print("="*50)

    try:
        # Initialize tester
        tester = MinneAppleModelTester(output_dir)
        tester.setup_models()

        # Run comparison
        comparison = tester.compare_models(image_path)

        print("📊 Comparison Results:")
        for model_name, result in comparison['models'].items():
            if 'error' not in result:
                print(f"\n{model_name.upper()}:")
                print(f"  🍎 Apples detected: {result['total_apples']}")
                print(f"  🎯 Avg confidence: {result['average_confidence']:.3f}")
                print(f"  ⚡ Inference time: {result['inference_time_ms']:.1f}ms")
                print(f"  🚀 FPS: {result['fps']:.1f}")
            else:
                print(f"\n{model_name.upper()}: ❌ {result['error']}")

        # Save comparison data
        comparison_file = Path(output_dir) / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"\n💾 Comparison results saved: {comparison_file}")

    except Exception as e:
        print(f"❌ Error in model comparison: {e}")


def demo_full_pipeline(image_path: str, output_dir: str = "results/pipeline_demo"):
    """
    Demo the complete pipeline with MinneApple weights

    Args:
        image_path: Path to test image
        output_dir: Output directory for results
    """
    print("\n🔧 FULL PIPELINE DEMO")
    print("="*50)

    try:
        # Initialize pipeline with MinneApple weights
        pipeline = ApplePipeline(
            config_path='config/config.yaml',
            detection_model_path=None  # Will use MinneApple weights from config
        )

        print("✅ Pipeline initialized with MinneApple integration")

        # Process image
        print(f"📍 Processing: {image_path}")
        result = pipeline.process_image(
            image=image_path,
            extract_quality=True,
            return_annotated=True
        )

        print(f"🍎 Total apples: {result.total_apples}")
        print(f"🎯 Quality score: {result.quality_score:.3f}")
        print(f"⚡ Processing time: {result.processing_time*1000:.1f}ms")
        print(f"📊 Quality distribution: {result.quality_distribution}")

        # Save annotated result
        if result.annotated_image is not None:
            import cv2
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            result_path = output_path / f"pipeline_result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(result_path), cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Pipeline result saved: {result_path}")

    except Exception as e:
        print(f"❌ Error in pipeline demo: {e}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='MinneApple Integration Demo')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch', 'presentation', 'comparison', 'pipeline', 'all'],
        default='single',
        help='Demo mode to run'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to test image (for single, comparison, pipeline modes)'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Directory with test images (for batch mode)'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        help='Path to results JSON file (for presentation mode)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/demo',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("🍎 MINNEAPPLE YOLOV8 INTEGRATION DEMO")
    print("="*60)

    if args.mode == 'single' or args.mode == 'all':
        if args.image:
            demo_single_image(args.image, args.output_dir)
        else:
            print("❌ --image required for single image demo")

    if args.mode == 'batch' or args.mode == 'all':
        if args.images_dir:
            demo_batch_processing(args.images_dir, args.output_dir + "_batch")
        else:
            print("❌ --images-dir required for batch demo")

    if args.mode == 'presentation' or args.mode == 'all':
        if args.results_file:
            demo_presentation_materials(args.results_file, args.output_dir + "_presentation")
        else:
            print("❌ --results-file required for presentation demo")

    if args.mode == 'comparison' or args.mode == 'all':
        if args.image:
            demo_model_comparison(args.image, args.output_dir + "_comparison")
        else:
            print("❌ --image required for comparison demo")

    if args.mode == 'pipeline' or args.mode == 'all':
        if args.image:
            demo_full_pipeline(args.image, args.output_dir + "_pipeline")
        else:
            print("❌ --image required for pipeline demo")

    print("\n✨ Demo completed!")


if __name__ == "__main__":
    main()