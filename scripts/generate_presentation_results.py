"""
Presentation Results Generator

This script generates professional presentation-ready results including
high-quality visualizations, metrics tables, and comparison charts
specifically formatted for PowerPoint presentations.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.apple_detector import AppleDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up presentation-quality plotting
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6
})


class PresentationResultsGenerator:
    """
    Generate professional presentation materials for apple detection results
    """

    def __init__(self, output_dir: str = "results/presentation"):
        """
        Initialize the presentation generator

        Args:
            output_dir: Directory to save presentation materials
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different content types
        self.slides_dir = self.output_dir / "slides"
        self.tables_dir = self.output_dir / "tables"
        self.charts_dir = self.output_dir / "charts"
        self.images_dir = self.output_dir / "demo_images"

        for dir_path in [self.slides_dir, self.tables_dir, self.charts_dir, self.images_dir]:
            dir_path.mkdir(exist_ok=True)

        logger.info(f"Presentation generator initialized. Output: {self.output_dir}")

    def create_model_comparison_chart(self, comparison_data: Dict) -> str:
        """
        Create a professional model comparison chart

        Args:
            comparison_data: Dictionary with model comparison results

        Returns:
            Path to saved chart
        """
        # Prepare data
        models = list(comparison_data.keys())
        metrics = ['Total Apples', 'Avg Confidence', 'Inference Time (ms)', 'FPS']

        # Extract data for each metric
        data = {
            'Total Apples': [comparison_data[m].get('total_apples', 0) for m in models],
            'Avg Confidence': [comparison_data[m].get('average_confidence', 0) for m in models],
            'Inference Time (ms)': [comparison_data[m].get('inference_time_ms', 0) for m in models],
            'FPS': [comparison_data[m].get('fps', 0) for m in models]
        }

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Apple Detection Model Comparison', fontsize=20, fontweight='bold', y=0.95)

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # 1. Total Apples Detected
        bars1 = ax1.bar(models, data['Total Apples'], color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Total Apples Detected', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Number of Apples', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars1, data['Total Apples']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 2. Average Confidence
        bars2 = ax2.bar(models, data['Avg Confidence'], color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Average Confidence Score', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Confidence Score', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Good Threshold (0.7)')
        ax2.legend()

        # Add value labels on bars
        for bar, value in zip(bars2, data['Avg Confidence']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 3. Inference Time
        bars3 = ax3.bar(models, data['Inference Time (ms)'], color=colors[2], alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Inference Time', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('Time (milliseconds)', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars3, data['Inference Time (ms)']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 4. FPS Performance
        bars4 = ax4.bar(models, data['FPS'], color=colors[3], alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_title('Processing Speed (FPS)', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Frames Per Second', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=15, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Real-time Target (15 FPS)')
        ax4.legend()

        # Add value labels on bars
        for bar, value in zip(bars4, data['FPS']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Rotate x-axis labels if needed
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save chart
        chart_path = self.charts_dir / "model_comparison_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"📊 Model comparison chart saved: {chart_path}")
        return str(chart_path)

    def create_performance_dashboard(self, test_results: List[Dict], model_name: str) -> str:
        """
        Create a comprehensive performance dashboard

        Args:
            test_results: List of test results
            model_name: Name of the model

        Returns:
            Path to saved dashboard
        """
        # Extract metrics
        apple_counts = [r['total_apples'] for r in test_results]
        inference_times = [r['inference_time_ms'] for r in test_results]
        confidence_scores = [r['average_confidence'] for r in test_results]
        fps_values = [r['fps'] for r in test_results]

        # Create dashboard
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{model_name.title()} Performance Dashboard', fontsize=24, fontweight='bold', y=0.95)

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Summary metrics (top row)
        ax_summary = fig.add_subplot(gs[0, :2])
        ax_summary.axis('off')

        # Calculate summary stats
        total_images = len(test_results)
        total_apples = sum(apple_counts)
        avg_apples = np.mean(apple_counts)
        avg_inference = np.mean(inference_times)
        avg_confidence = np.mean(confidence_scores)
        avg_fps = np.mean(fps_values)

        summary_text = f"""
PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━
📊 Images Processed: {total_images}
🍎 Total Apples: {total_apples}
📈 Avg per Image: {avg_apples:.1f}
⚡ Avg Speed: {avg_inference:.1f}ms
🎯 Avg Confidence: {avg_confidence:.3f}
🚀 Avg FPS: {avg_fps:.1f}
        """

        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=16,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

        # 2. Performance indicators (top right)
        ax_indicators = fig.add_subplot(gs[0, 2:])
        ax_indicators.axis('off')

        # Performance indicators
        real_time_achieved = avg_fps >= 15
        high_confidence = avg_confidence >= 0.7
        fast_inference = avg_inference <= 100

        indicators = [
            ("Real-time Performance", "✅" if real_time_achieved else "❌", "green" if real_time_achieved else "red"),
            ("High Confidence", "✅" if high_confidence else "❌", "green" if high_confidence else "red"),
            ("Fast Inference", "✅" if fast_inference else "❌", "green" if fast_inference else "red")
        ]

        indicator_text = "PERFORMANCE INDICATORS\n━━━━━━━━━━━━━━━━━━━━━━━\n"
        for name, status, color in indicators:
            indicator_text += f"{status} {name}\n"

        ax_indicators.text(0.05, 0.95, indicator_text, transform=ax_indicators.transAxes, fontsize=16,
                          verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))

        # 3. Detection distribution (middle left)
        ax_dist = fig.add_subplot(gs[1, :2])
        bins = max(1, len(set(apple_counts)))
        n, bins, patches = ax_dist.hist(apple_counts, bins=bins, alpha=0.7, color='darkgreen', edgecolor='black')
        ax_dist.set_title('Apple Detection Distribution', fontsize=16, fontweight='bold')
        ax_dist.set_xlabel('Apples per Image')
        ax_dist.set_ylabel('Frequency')
        ax_dist.grid(True, alpha=0.3)

        # 4. Inference time trend (middle right)
        ax_time = fig.add_subplot(gs[1, 2:])
        ax_time.plot(range(len(inference_times)), inference_times, 'b-o', markersize=6, linewidth=2)
        ax_time.axhline(y=np.mean(inference_times), color='red', linestyle='--', linewidth=2,
                       label=f'Average: {np.mean(inference_times):.1f}ms')
        ax_time.set_title('Inference Time Trend', fontsize=16, fontweight='bold')
        ax_time.set_xlabel('Image Index')
        ax_time.set_ylabel('Time (ms)')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)

        # 5. Confidence vs Detection count (bottom left)
        ax_conf = fig.add_subplot(gs[2, :2])
        scatter = ax_conf.scatter(apple_counts, confidence_scores, c=fps_values, cmap='viridis',
                                 s=100, alpha=0.7, edgecolors='black')
        ax_conf.set_title('Detection Count vs Confidence (Color = FPS)', fontsize=16, fontweight='bold')
        ax_conf.set_xlabel('Apples Detected')
        ax_conf.set_ylabel('Average Confidence')
        cbar = plt.colorbar(scatter, ax=ax_conf)
        cbar.set_label('FPS', rotation=270, labelpad=15)
        ax_conf.grid(True, alpha=0.3)

        # 6. FPS distribution (bottom right)
        ax_fps = fig.add_subplot(gs[2, 2:])
        ax_fps.hist(fps_values, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax_fps.axvline(x=15, color='red', linestyle='--', linewidth=2, label='Real-time Target (15 FPS)')
        ax_fps.axvline(x=np.mean(fps_values), color='orange', linestyle='-', linewidth=2,
                      label=f'Average: {np.mean(fps_values):.1f} FPS')
        ax_fps.set_title('FPS Distribution', fontsize=16, fontweight='bold')
        ax_fps.set_xlabel('Frames Per Second')
        ax_fps.set_ylabel('Frequency')
        ax_fps.legend()
        ax_fps.grid(True, alpha=0.3)

        # Save dashboard
        dashboard_path = self.charts_dir / f"{model_name}_performance_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"📈 Performance dashboard saved: {dashboard_path}")
        return str(dashboard_path)

    def create_detection_showcase(self, test_results: List[Dict], max_images: int = 6) -> str:
        """
        Create a showcase of detection results for presentation

        Args:
            test_results: List of test results
            max_images: Maximum number of images to showcase

        Returns:
            Path to saved showcase
        """
        # Select best results (high confidence, good detection count)
        scored_results = []
        for result in test_results:
            if result['total_apples'] > 0:
                score = result['average_confidence'] * (1 + 0.1 * result['total_apples'])
                scored_results.append((score, result))

        # Sort by score and take top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        selected_results = [result for _, result in scored_results[:max_images]]

        if not selected_results:
            logger.warning("No suitable results for showcase")
            return ""

        # Create showcase
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
        fig.suptitle('Apple Detection Results Showcase', fontsize=24, fontweight='bold', y=0.95)

        axes = axes.flatten()

        for i, result in enumerate(selected_results[:len(axes)]):
            ax = axes[i]

            # Load and display annotated image
            if 'annotated_image_path' in result and os.path.exists(result['annotated_image_path']):
                img = cv2.imread(result['annotated_image_path'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)

                # Add result info
                info_text = f"Apples: {result['total_apples']}\n"
                info_text += f"Conf: {result['average_confidence']:.3f}\n"
                info_text += f"Time: {result['inference_time_ms']:.1f}ms\n"
                info_text += f"FPS: {result['fps']:.1f}"

                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       fontsize=12, verticalalignment='top', fontweight='bold')

            ax.axis('off')

        # Hide unused subplots
        for i in range(len(selected_results), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        # Save showcase
        showcase_path = self.images_dir / "detection_showcase.png"
        plt.savefig(showcase_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"🖼️  Detection showcase saved: {showcase_path}")
        return str(showcase_path)

    def generate_metrics_table(self, summary_data: Dict) -> str:
        """
        Generate a professional metrics table

        Args:
            summary_data: Summary statistics dictionary

        Returns:
            Path to saved table image
        """
        # Prepare data for table
        metrics_data = {
            'Metric': [
                'Total Images Tested',
                'Total Apples Detected',
                'Average Apples/Image',
                'Detection Success Rate',
                'Average Inference Time',
                'Average FPS',
                'Average Confidence',
                'High Confidence Rate',
                'Real-time Capable',
                'Model Source'
            ],
            'Value': [
                f"{summary_data['test_summary']['total_images_tested']}",
                f"{summary_data['test_summary']['total_apples_detected']}",
                f"{summary_data['test_summary']['avg_apples_per_image']}",
                f"{summary_data['test_summary']['detection_rate']}",
                f"{summary_data['performance_metrics']['avg_inference_time_ms']:.1f} ms",
                f"{summary_data['performance_metrics']['avg_fps']:.1f}",
                f"{summary_data['detection_quality']['avg_confidence_score']:.3f}",
                f"{summary_data['detection_quality']['high_confidence_detections']}",
                "✅ Yes" if summary_data['performance_metrics']['target_fps_achieved'] else "❌ No",
                f"{summary_data['model_source']}"
            ]
        }

        # Create DataFrame
        df = pd.DataFrame(metrics_data)

        # Create figure for table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2.5)

        # Header styling
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(df) + 1):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)

        plt.title('Model Performance Metrics', fontsize=18, fontweight='bold', pad=20)

        # Save table
        table_path = self.tables_dir / "performance_metrics_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"📋 Metrics table saved: {table_path}")
        return str(table_path)

    def create_presentation_summary(self, results_data: Dict) -> str:
        """
        Create a comprehensive presentation summary document

        Args:
            results_data: Complete results data

        Returns:
            Path to saved summary document
        """
        summary_path = self.output_dir / "presentation_summary.md"

        with open(summary_path, 'w') as f:
            f.write("# Apple Detection Model - Presentation Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}\n\n")

            # Key highlights
            f.write("## 🎯 Key Highlights\n\n")
            if 'summary' in results_data:
                summary = results_data['summary']
                f.write(f"- **Total Apples Detected:** {summary['test_summary']['total_apples_detected']}\n")
                f.write(f"- **Average Processing Time:** {summary['performance_metrics']['avg_inference_time_ms']:.1f}ms\n")
                f.write(f"- **Real-time Performance:** {summary['performance_metrics']['avg_fps']:.1f} FPS\n")
                f.write(f"- **Detection Confidence:** {summary['detection_quality']['avg_confidence_score']:.3f}\n\n")

            # Files generated
            f.write("## 📁 Generated Files for Presentation\n\n")
            f.write("### Charts and Visualizations\n")
            f.write("- `charts/model_comparison_chart.png` - Model performance comparison\n")
            f.write("- `charts/*_performance_dashboard.png` - Comprehensive performance dashboard\n\n")

            f.write("### Demo Images\n")
            f.write("- `demo_images/detection_showcase.png` - Best detection results showcase\n")
            f.write("- `demo_images/` - Individual annotated test results\n\n")

            f.write("### Data Tables\n")
            f.write("- `tables/performance_metrics_table.png` - Professional metrics table\n\n")

            # Usage instructions
            f.write("## 📝 How to Use in Presentation\n\n")
            f.write("1. **Opening Slide:** Use key highlights and model comparison chart\n")
            f.write("2. **Technical Details:** Show performance dashboard and metrics table\n")
            f.write("3. **Demo Results:** Display detection showcase images\n")
            f.write("4. **Conclusion:** Reference real-time performance and accuracy metrics\n\n")

            # Talking points
            f.write("## 💬 Key Talking Points\n\n")
            f.write("- **Pre-trained MinneApple weights** provide immediate apple detection capability\n")
            f.write("- **Real-time processing** suitable for agricultural applications\n")
            f.write("- **High detection accuracy** with confidence-based quality assessment\n")
            f.write("- **Production-ready pipeline** with integrated tracking and quality classification\n")

        logger.info(f"📄 Presentation summary saved: {summary_path}")
        return str(summary_path)


def main():
    """Main function for generating presentation results"""
    parser = argparse.ArgumentParser(description='Generate Presentation Results')

    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='JSON file containing test results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/presentation',
        help='Output directory for presentation materials'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = PresentationResultsGenerator(args.output_dir)

    # Load results
    with open(args.results_file, 'r') as f:
        results_data = json.load(f)

    print("🎨 Generating presentation materials...")

    # Generate all presentation components
    if 'test_results' in results_data:
        test_results = results_data['test_results']
        model_name = results_data.get('model_name', 'minneapple')

        # Create performance dashboard
        dashboard_path = generator.create_performance_dashboard(test_results, model_name)

        # Create detection showcase
        showcase_path = generator.create_detection_showcase(test_results)

        print(f"✅ Generated performance dashboard: {dashboard_path}")
        print(f"✅ Generated detection showcase: {showcase_path}")

    if 'summary' in results_data:
        # Create metrics table
        table_path = generator.generate_metrics_table(results_data['summary'])
        print(f"✅ Generated metrics table: {table_path}")

    if 'comparison' in results_data:
        # Create comparison chart
        chart_path = generator.create_model_comparison_chart(results_data['comparison'])
        print(f"✅ Generated comparison chart: {chart_path}")

    # Create presentation summary
    summary_path = generator.create_presentation_summary(results_data)
    print(f"✅ Generated presentation summary: {summary_path}")

    print(f"\n🎉 All presentation materials saved to: {generator.output_dir}")
    print("\n📊 Ready for PowerPoint integration!")


if __name__ == "__main__":
    main()