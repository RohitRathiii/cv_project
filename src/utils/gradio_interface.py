"""
Gradio Web Interface for Apple Detection and Quality Grading Pipeline

This module provides a user-friendly web interface for demonstrating the
apple detection and quality grading capabilities using Gradio.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import cv2
import gradio as gr
import pandas as pd
from typing import Tuple, Optional
import yaml
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.apple_pipeline import ApplePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppleGradioInterface:
    """
    Gradio interface for Apple Detection and Quality Grading Pipeline
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        detection_model_path: Optional[str] = None,
        quality_model_path: Optional[str] = None
    ):
        """
        Initialize Gradio interface
        
        Args:
            config_path: Path to configuration file
            detection_model_path: Path to detection model weights
            quality_model_path: Path to quality model weights
        """
        self.pipeline = None
        self.config_path = config_path
        self.detection_model_path = detection_model_path
        self.quality_model_path = quality_model_path
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        # Interface state
        self.processing_history = []
        
    def _initialize_pipeline(self):
        """Initialize the apple pipeline"""
        try:
            self.pipeline = ApplePipeline(
                config_path=self.config_path,
                detection_model_path=self.detection_model_path,
                quality_model_path=self.quality_model_path
            )
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            self.pipeline = None
    
    def process_image(
        self,
        image: np.ndarray,
        extract_quality: bool = True,
        confidence_threshold: float = 0.3
    ) -> Tuple[np.ndarray, str, str]:
        """
        Process uploaded image through the pipeline
        
        Args:
            image: Input image from Gradio
            extract_quality: Whether to perform quality assessment
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Tuple of (annotated_image, results_text, detailed_report)
        """
        if self.pipeline is None:
            return image, "Error: Pipeline not initialized", ""
        
        try:
            # Update confidence threshold
            if hasattr(self.pipeline.detector, 'conf_threshold'):
                self.pipeline.detector.conf_threshold = confidence_threshold
            
            # Process image
            start_time = time.time()
            result = self.pipeline.process_image(
                image=image,
                extract_quality=extract_quality,
                return_annotated=True
            )
            processing_time = time.time() - start_time
            
            # Create results summary
            results_text = self._format_results_summary(result, processing_time)
            
            # Create detailed report
            detailed_report = self._format_detailed_report(result)
            
            # Store in history
            self.processing_history.append({
                'timestamp': time.time(),
                'result': result,
                'processing_time': processing_time
            })
            
            return result.annotated_image, results_text, detailed_report
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return image, f"Error: {str(e)}", ""
    
    def process_video(
        self,
        video_file: str,
        extract_quality: bool = True,
        confidence_threshold: float = 0.3,
        frame_skip: int = 1
    ) -> Tuple[str, str]:
        """
        Process uploaded video through the pipeline
        
        Args:
            video_file: Path to uploaded video file
            extract_quality: Whether to perform quality assessment
            confidence_threshold: Detection confidence threshold
            frame_skip: Process every nth frame
            
        Returns:
            Tuple of (output_video_path, results_summary)
        """
        if self.pipeline is None:
            return None, "Error: Pipeline not initialized"
        
        if not video_file:
            return None, "Please upload a video file"
        
        try:
            # Update confidence threshold
            if hasattr(self.pipeline.detector, 'conf_threshold'):
                self.pipeline.detector.conf_threshold = confidence_threshold
            
            # Create output path
            output_dir = "gradio_outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_video_path = os.path.join(output_dir, f"processed_video_{int(time.time())}.mp4")
            
            # Process video
            results = self.pipeline.process_video(
                video_source=video_file,
                extract_quality=extract_quality,
                save_video=True,
                output_path=output_video_path,
                frame_skip=frame_skip
            )
            
            # Generate summary report
            summary_report = self._format_video_summary(results)
            
            return output_video_path, summary_report
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def _format_results_summary(self, result, processing_time: float) -> str:
        """Format results into a readable summary"""
        summary = f"""
🍎 **Apple Detection Results**

**Detection Summary:**
• Total Apples Detected: {result.total_apples}
• Processing Time: {processing_time:.2f} seconds
• Average Confidence: {np.mean(result.confidence_scores):.3f if result.confidence_scores else 0:.3f}

**Quality Assessment:**
• Good Apples: {result.quality_distribution['good']} ({result.quality_distribution['good']/max(result.total_apples, 1)*100:.1f}%)
• Minor Defects: {result.quality_distribution['minor_defect']} ({result.quality_distribution['minor_defect']/max(result.total_apples, 1)*100:.1f}%)
• Major Defects: {result.quality_distribution['major_defect']} ({result.quality_distribution['major_defect']/max(result.total_apples, 1)*100:.1f}%)
• Overall Quality Score: {result.quality_score:.3f}/1.000

**Performance:**
• FPS: {1/processing_time:.1f}
"""
        return summary
    
    def _format_detailed_report(self, result) -> str:
        """Format detailed results report"""
        if not result.detailed_results:
            return "No detailed results available"
        
        detailed = result.detailed_results
        
        report = f"""
**Detailed Analysis Report**

**Image Information:**
• Image Shape: {detailed.get('image_shape', 'Unknown')}
• Image Path: {detailed.get('image_path', 'Uploaded image')}

**Detection Details:**
• Number of Detections: {len(detailed.get('detections', {}).get('boxes', []))}
• Confidence Scores: {[f"{score:.3f}" for score in detailed.get('detections', {}).get('confidence_scores', [])]}

**Quality Assessment Details:**
"""
        
        quality_assessments = detailed.get('quality_assessments', [])
        for i, assessment in enumerate(quality_assessments):
            report += f"• Apple {i+1}: {assessment['predicted_class']} (confidence: {assessment['confidence']:.3f})\n"
        
        processing_breakdown = detailed.get('processing_breakdown', {})
        report += f"""
**Performance Breakdown:**
• Detection Time: {processing_breakdown.get('detection_time', 0):.3f}s
• Quality Assessment Time: {processing_breakdown.get('quality_time', 0):.3f}s
• Total Processing Time: {processing_breakdown.get('total_time', 0):.3f}s
"""
        
        return report
    
    def _format_video_summary(self, results) -> str:
        """Format video processing summary"""
        if not results:
            return "No results available"
        
        total_frames = len(results)
        total_apples = sum(r.total_apples for r in results)
        unique_apples = results[-1].unique_apples if results else 0
        
        # Calculate quality distribution
        total_quality = {'good': 0, 'minor_defect': 0, 'major_defect': 0}
        for result in results:
            for quality_class, count in result.quality_distribution.items():
                total_quality[quality_class] += count
        
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        summary = f"""
🎥 **Video Processing Results**

**Video Summary:**
• Total Frames Processed: {total_frames}
• Total Apple Detections: {total_apples}
• Unique Apples Tracked: {unique_apples}
• Average Apples per Frame: {total_apples/total_frames:.1f}

**Quality Assessment:**
• Total Good Apples: {total_quality['good']}
• Total Minor Defects: {total_quality['minor_defect']}
• Total Major Defects: {total_quality['major_defect']}

**Performance:**
• Average Processing Time: {avg_processing_time:.3f}s per frame
• Average FPS: {1/avg_processing_time:.1f}

**Tracking Efficiency:**
• Detection to Track Ratio: {unique_apples/max(total_apples, 1):.3f}
"""
        return summary
    
    def get_processing_history(self) -> str:
        """Get processing history summary"""
        if not self.processing_history:
            return "No processing history available"
        
        history_text = "**Processing History:**\n\n"
        
        for i, entry in enumerate(self.processing_history[-5:], 1):  # Show last 5 entries
            result = entry['result']
            timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
            
            history_text += f"""
**{i}. {timestamp}**
• Apples: {result.total_apples}
• Quality Score: {result.quality_score:.3f}
• Processing Time: {entry['processing_time']:.2f}s
---
"""
        
        return history_text
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="Apple Detection and Quality Grading",
            theme=gr.themes.Soft(),
            css="""
            .container { max-width: 1200px; margin: auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .results-panel { background-color: #f8f9fa; padding: 20px; border-radius: 10px; }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🍎 Apple Detection and Quality Grading Pipeline</h1>
                <p>Upload images or videos to detect apples and assess their quality using AI</p>
            </div>
            """)
            
            with gr.Tabs():
                # Image Processing Tab
                with gr.TabItem("📷 Image Processing"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                label="Upload Apple Image",
                                type="numpy",
                                height=400
                            )
                            
                            with gr.Row():
                                quality_checkbox = gr.Checkbox(
                                    label="Enable Quality Assessment",
                                    value=True
                                )
                                confidence_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.3,
                                    step=0.1,
                                    label="Confidence Threshold"
                                )
                            
                            process_btn = gr.Button(
                                "🔍 Process Image",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            output_image = gr.Image(
                                label="Processed Result",
                                height=400
                            )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            results_text = gr.Markdown(
                                label="Results Summary",
                                value="Upload an image to see results"
                            )
                        
                        with gr.Column(scale=1):
                            detailed_report = gr.Markdown(
                                label="Detailed Report",
                                value="Detailed analysis will appear here"
                            )
                
                # Video Processing Tab
                with gr.TabItem("🎥 Video Processing"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            video_input = gr.File(
                                label="Upload Video File",
                                file_types=[".mp4", ".avi", ".mov", ".mkv"]
                            )
                            
                            with gr.Row():
                                video_quality_checkbox = gr.Checkbox(
                                    label="Enable Quality Assessment",
                                    value=True
                                )
                                video_confidence_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.3,
                                    step=0.1,
                                    label="Confidence Threshold"
                                )
                            
                            frame_skip_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1,
                                label="Frame Skip (process every nth frame)"
                            )
                            
                            process_video_btn = gr.Button(
                                "🎬 Process Video",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            output_video = gr.Video(
                                label="Processed Video"
                            )
                            
                            video_results = gr.Markdown(
                                label="Video Analysis Results",
                                value="Upload a video to see results"
                            )
                
                # History Tab
                with gr.TabItem("📊 Processing History"):
                    history_display = gr.Markdown(
                        label="Processing History",
                        value="No processing history available"
                    )
                    
                    refresh_history_btn = gr.Button(
                        "🔄 Refresh History",
                        variant="secondary"
                    )
                
                # About Tab
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ## About This Application
                    
                    This application demonstrates an end-to-end computer vision pipeline for apple detection and quality grading.
                    
                    ### Features:
                    - **Apple Detection**: Uses YOLOv8 for accurate apple detection in images and videos
                    - **Quality Assessment**: Employs MobileNetV3 to classify apple quality (Good, Minor Defect, Major Defect)
                    - **Multi-Object Tracking**: Tracks apples across video frames for accurate counting
                    - **Real-time Processing**: Optimized for efficient inference
                    
                    ### Model Architecture:
                    - **Detection**: YOLOv8n (3.2M parameters)
                    - **Quality Classification**: MobileNetV3-Small (2.9M parameters)
                    - **Tracking**: DeepSORT with Kalman filtering
                    
                    ### Performance Targets:
                    - Detection mAP > 90%
                    - Quality Classification Accuracy > 92%
                    - Real-time inference (>15 FPS)
                    
                    ### Usage Tips:
                    - For best results, use images with clear apple visibility
                    - Adjust confidence threshold based on your requirements
                    - Enable quality assessment for detailed analysis
                    - Use frame skipping for faster video processing
                    """)
            
            # Event handlers
            process_btn.click(
                fn=self.process_image,
                inputs=[image_input, quality_checkbox, confidence_slider],
                outputs=[output_image, results_text, detailed_report],
                show_progress=True
            )
            
            process_video_btn.click(
                fn=self.process_video,
                inputs=[video_input, video_quality_checkbox, video_confidence_slider, frame_skip_slider],
                outputs=[output_video, video_results],
                show_progress=True
            )
            
            refresh_history_btn.click(
                fn=self.get_processing_history,
                outputs=[history_display]
            )
        
        return interface
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        debug: bool = False
    ):
        """
        Launch the Gradio interface
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            debug: Enable debug mode
        """
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=debug,
            show_error=True
        )


def main():
    """Main function to launch the Gradio interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Apple Detection Gradio Interface')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--detection-model', type=str, help='Path to detection model weights')
    parser.add_argument('--quality-model', type=str, help='Path to quality model weights')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server hostname')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and launch interface
    app = AppleGradioInterface(
        config_path=args.config,
        detection_model_path=args.detection_model,
        quality_model_path=args.quality_model
    )
    
    app.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()