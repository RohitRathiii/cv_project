"""
Evaluation Metrics and Reporting Utilities

This module provides comprehensive evaluation metrics for the apple detection
and quality grading pipeline, including visualization and reporting tools.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, average_precision_score
)
import json
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """
    Evaluation utilities for object detection models
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize detection evaluator
        
        Args:
            iou_threshold: IoU threshold for positive detections
        """
        self.iou_threshold = iou_threshold
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_detections(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        confidence_thresholds: List[float] = None
    ) -> Dict:
        """
        Evaluate detection results against ground truth
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            confidence_thresholds: List of confidence thresholds to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if confidence_thresholds is None:
            confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {}
        
        for conf_thresh in confidence_thresholds:
            # Filter predictions by confidence
            filtered_preds = []
            for pred in predictions:
                filtered_boxes = []
                filtered_scores = []
                
                for box, score in zip(pred['boxes'], pred['confidence_scores']):
                    if score >= conf_thresh:
                        filtered_boxes.append(box)
                        filtered_scores.append(score)
                
                filtered_preds.append({
                    'boxes': filtered_boxes,
                    'confidence_scores': filtered_scores,
                    'total_apples': len(filtered_boxes)
                })
            
            # Calculate metrics
            metrics = self._calculate_metrics(filtered_preds, ground_truths)
            results[f'conf_{conf_thresh}'] = metrics
        
        # Calculate mAP
        map_score = self._calculate_map(predictions, ground_truths)
        results['mAP'] = map_score
        
        return results
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """Calculate precision, recall, F1, and counting metrics"""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        counting_errors = []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            gt_boxes = gt['boxes']
            
            # Convert to xyxy format if needed
            pred_xyxy = []
            for box in pred_boxes:
                if 'x1' in box:
                    pred_xyxy.append([box['x1'], box['y1'], box['x2'], box['y2']])
                else:
                    pred_xyxy.append(box)
            
            gt_xyxy = []
            for box in gt_boxes:
                if 'x1' in box:
                    gt_xyxy.append([box['x1'], box['y1'], box['x2'], box['y2']])
                else:
                    gt_xyxy.append(box)
            
            # Match predictions to ground truth
            matched_gt = set()
            tp = 0
            
            for pred_box in pred_xyxy:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_xyxy):
                    if i in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= self.iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
            
            fp = len(pred_xyxy) - tp
            fn = len(gt_xyxy) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Counting error
            counting_error = abs(len(pred_xyxy) - len(gt_xyxy))
            counting_errors.append(counting_error)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Counting metrics
        mae = np.mean(counting_errors)
        mape = np.mean([err / max(len(gt['boxes']), 1) for err, gt in zip(counting_errors, ground_truths)]) * 100
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mae': mae,
            'mape': mape,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    
    def _calculate_map(self, predictions: List[Dict], ground_truths: List[Dict]) -> float:
        """Calculate mean Average Precision (mAP)"""
        # Simplified mAP calculation
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []
        
        for iou_thresh in iou_thresholds:
            self.iou_threshold = iou_thresh
            metrics = self._calculate_metrics(predictions, ground_truths)
            aps.append(metrics['precision'])
        
        return np.mean(aps)


class ClassificationEvaluator:
    """
    Evaluation utilities for classification models
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize classification evaluator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ['good', 'minor_defect', 'major_defect']
    
    def evaluate_classification(
        self,
        predictions: List[int],
        ground_truths: List[int],
        prediction_probabilities: Optional[List[List[float]]] = None
    ) -> Dict:
        """
        Evaluate classification results
        
        Args:
            predictions: Predicted class indices
            ground_truths: Ground truth class indices
            prediction_probabilities: Prediction probabilities for each class
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision_macro = precision_score(ground_truths, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(ground_truths, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(ground_truths, predictions, average='macro', zero_division=0)
        
        precision_weighted = precision_score(ground_truths, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(ground_truths, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(ground_truths, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(ground_truths, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(ground_truths, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        
        # Classification report
        report = classification_report(
            ground_truths, 
            predictions, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': precision_per_class[i] if i < len(precision_per_class) else 0,
                    'recall': recall_per_class[i] if i < len(recall_per_class) else 0,
                    'f1': f1_per_class[i] if i < len(f1_per_class) else 0
                }
                for i in range(len(self.class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Add AUC if probabilities provided
        if prediction_probabilities is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc_scores = {}
                
                for i, class_name in enumerate(self.class_names):
                    if len(np.unique(ground_truths)) > 1:  # Need at least 2 classes
                        y_true_binary = (np.array(ground_truths) == i).astype(int)
                        y_prob = np.array(prediction_probabilities)[:, i]
                        
                        if len(np.unique(y_true_binary)) > 1:  # Need positive and negative samples
                            auc = roc_auc_score(y_true_binary, y_prob)
                            auc_scores[class_name] = auc
                
                results['auc_scores'] = auc_scores
                
            except Exception as e:
                logger.warning(f"Could not calculate AUC scores: {str(e)}")
        
        return results


class PipelineEvaluator:
    """
    Comprehensive pipeline evaluation utilities
    """
    
    def __init__(self):
        """Initialize pipeline evaluator"""
        self.detection_evaluator = DetectionEvaluator()
        self.classification_evaluator = ClassificationEvaluator()
    
    def evaluate_pipeline(
        self,
        pipeline_results: List[Dict],
        ground_truth_data: List[Dict]
    ) -> Dict:
        """
        Evaluate complete pipeline performance
        
        Args:
            pipeline_results: Results from pipeline processing
            ground_truth_data: Ground truth annotations
            
        Returns:
            Comprehensive evaluation results
        """
        # Extract detection results
        detection_predictions = []
        detection_ground_truths = []
        
        # Extract classification results
        classification_predictions = []
        classification_ground_truths = []
        classification_probabilities = []
        
        for pipeline_result, gt_data in zip(pipeline_results, ground_truth_data):
            # Detection data
            detection_predictions.append({
                'boxes': pipeline_result.get('detections', {}).get('boxes', []),
                'confidence_scores': pipeline_result.get('detections', {}).get('confidence_scores', []),
                'total_apples': pipeline_result.get('total_apples', 0)
            })
            
            detection_ground_truths.append({
                'boxes': gt_data.get('detection_boxes', []),
                'total_apples': len(gt_data.get('detection_boxes', []))
            })
            
            # Classification data
            quality_results = pipeline_result.get('quality_assessments', [])
            gt_quality = gt_data.get('quality_labels', [])
            
            for quality_result, gt_label in zip(quality_results, gt_quality):
                predicted_class = quality_result['predicted_class']
                class_names = ['good', 'minor_defect', 'major_defect']
                
                if predicted_class in class_names:
                    classification_predictions.append(class_names.index(predicted_class))
                    classification_ground_truths.append(gt_label)
                    
                    # Extract probabilities
                    probs = [
                        quality_result['probabilities'].get(class_name, 0.0)
                        for class_name in class_names
                    ]
                    classification_probabilities.append(probs)
        
        # Evaluate detection
        detection_results = {}
        if detection_predictions and detection_ground_truths:
            detection_results = self.detection_evaluator.evaluate_detections(
                detection_predictions, detection_ground_truths
            )
        
        # Evaluate classification
        classification_results = {}
        if classification_predictions and classification_ground_truths:
            classification_results = self.classification_evaluator.evaluate_classification(
                classification_predictions,
                classification_ground_truths,
                classification_probabilities if classification_probabilities else None
            )
        
        # Calculate pipeline-specific metrics
        pipeline_metrics = self._calculate_pipeline_metrics(pipeline_results)
        
        return {
            'detection_evaluation': detection_results,
            'classification_evaluation': classification_results,
            'pipeline_metrics': pipeline_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_pipeline_metrics(self, pipeline_results: List[Dict]) -> Dict:
        """Calculate pipeline-specific performance metrics"""
        processing_times = [result.get('processing_time', 0) for result in pipeline_results]
        
        metrics = {
            'total_images_processed': len(pipeline_results),
            'average_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'average_fps': 1.0 / np.mean(processing_times) if processing_times and np.mean(processing_times) > 0 else 0,
            'total_apples_detected': sum(result.get('total_apples', 0) for result in pipeline_results)
        }
        
        return metrics


class ReportGenerator:
    """
    Generate comprehensive evaluation reports with visualizations
    """
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_detection_report(
        self,
        evaluation_results: Dict,
        save_plots: bool = True
    ) -> str:
        """Generate detection evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"detection_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._generate_detection_html(evaluation_results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        if save_plots:
            self._save_detection_plots(evaluation_results, timestamp)
        
        logger.info(f"Detection report saved to {report_path}")
        return str(report_path)
    
    def generate_classification_report(
        self,
        evaluation_results: Dict,
        save_plots: bool = True
    ) -> str:
        """Generate classification evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"classification_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._generate_classification_html(evaluation_results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        if save_plots:
            self._save_classification_plots(evaluation_results, timestamp)
        
        logger.info(f"Classification report saved to {report_path}")
        return str(report_path)
    
    def _generate_detection_html(self, results: Dict) -> str:
        """Generate HTML content for detection report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Apple Detection Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-value {{ font-weight: bold; color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🍎 Apple Detection Evaluation Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add metrics tables
        if 'mAP' in results:
            html += f'<div class="metric">Overall mAP: <span class="metric-value">{results["mAP"]:.4f}</span></div>'
        
        # Confidence threshold results
        html += "<h2>Performance by Confidence Threshold</h2><table><tr><th>Confidence</th><th>Precision</th><th>Recall</th><th>F1</th><th>MAE</th></tr>"
        
        for key, metrics in results.items():
            if key.startswith('conf_'):
                conf = key.replace('conf_', '')
                html += f"""
                <tr>
                    <td>{conf}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1']:.4f}</td>
                    <td>{metrics['mae']:.2f}</td>
                </tr>
                """
        
        html += "</table></body></html>"
        return html
    
    def _generate_classification_html(self, results: Dict) -> str:
        """Generate HTML content for classification report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Apple Quality Classification Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-value {{ font-weight: bold; color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #27ae60; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🍎 Apple Quality Classification Evaluation Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Overall metrics
        html += f"""
        <h2>Overall Performance</h2>
        <div class="metric">Accuracy: <span class="metric-value">{results.get('accuracy', 0):.4f}</span></div>
        <div class="metric">Macro F1: <span class="metric-value">{results.get('f1_macro', 0):.4f}</span></div>
        <div class="metric">Macro Precision: <span class="metric-value">{results.get('precision_macro', 0):.4f}</span></div>
        <div class="metric">Macro Recall: <span class="metric-value">{results.get('recall_macro', 0):.4f}</span></div>
        """
        
        # Per-class metrics
        html += "<h2>Per-Class Performance</h2><table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>"
        
        per_class = results.get('per_class_metrics', {})
        for class_name, metrics in per_class.items():
            html += f"""
            <tr>
                <td>{class_name}</td>
                <td>{metrics['precision']:.4f}</td>
                <td>{metrics['recall']:.4f}</td>
                <td>{metrics['f1']:.4f}</td>
            </tr>
            """
        
        html += "</table></body></html>"
        return html
    
    def _save_detection_plots(self, results: Dict, timestamp: str):
        """Save detection evaluation plots"""
        # Precision-Recall curve by confidence threshold
        plt.figure(figsize=(10, 6))
        
        conf_thresholds = []
        precisions = []
        recalls = []
        
        for key, metrics in results.items():
            if key.startswith('conf_'):
                conf = float(key.replace('conf_', ''))
                conf_thresholds.append(conf)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
        
        plt.subplot(1, 2, 1)
        plt.plot(conf_thresholds, precisions, 'b-o', label='Precision')
        plt.plot(conf_thresholds, recalls, 'r-s', label='Recall')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(recalls, precisions, 'g-o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / f"detection_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_classification_plots(self, results: Dict, timestamp: str):
        """Save classification evaluation plots"""
        # Confusion matrix
        if 'confusion_matrix' in results:
            plt.figure(figsize=(8, 6))
            cm = np.array(results['confusion_matrix'])
            class_names = ['Good', 'Minor Defect', 'Major Defect']
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plot_path = self.output_dir / f"confusion_matrix_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Example usage
    evaluator = PipelineEvaluator()
    
    # Example evaluation
    # results = evaluator.evaluate_pipeline(pipeline_results, ground_truth_data)
    # print(f"Evaluation completed: {results}")