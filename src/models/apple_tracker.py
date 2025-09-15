"""
Apple Tracking Module using DeepSORT

This module implements the AppleTracker class for multi-object tracking
of apples across video frames using DeepSORT algorithm.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import cv2
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSORT not available. Install with: pip install deep-sort-realtime")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection data structure"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # Apple class


@dataclass
class Track:
    """Track data structure"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    age: int = 0
    hit_streak: int = 0
    time_since_update: int = 0


class SimpleTracker:
    """
    Simple tracking implementation as fallback when DeepSORT is not available
    Uses IoU-based tracking with Kalman filter for motion prediction
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize Simple Tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections needed to confirm track
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.track_count = 0
        self.frame_count = 0
        
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of current frame detections
            
        Returns:
            List of updated tracks
        """
        self.frame_count += 1
        
        # Convert detections to simple format
        det_boxes = [det.bbox for det in detections]
        det_scores = [det.confidence for det in detections]
        
        # Association matrix
        if len(self.tracks) > 0 and len(det_boxes) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(det_boxes)))
            
            for i, track in enumerate(self.tracks):
                for j, det_box in enumerate(det_boxes):
                    iou_matrix[i, j] = self._calculate_iou(track.bbox, det_box)
            
            # Simple greedy assignment
            matched_indices = []
            for i in range(len(self.tracks)):
                if len(det_boxes) == 0:
                    break
                best_match = np.argmax(iou_matrix[i])
                if iou_matrix[i, best_match] > self.iou_threshold:
                    matched_indices.append((i, best_match))
                    iou_matrix[:, best_match] = 0  # Remove matched detection
        else:
            matched_indices = []
        
        # Update matched tracks
        matched_tracks = set()
        matched_dets = set()
        
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].bbox = det_boxes[det_idx]
            self.tracks[track_idx].confidence = det_scores[det_idx]
            self.tracks[track_idx].hit_streak += 1
            self.tracks[track_idx].time_since_update = 0
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_dets:
                new_track = Track(
                    track_id=self.track_count,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    hit_streak=1,
                    time_since_update=0
                )
                self.tracks.append(new_track)
                self.track_count += 1
        
        # Update unmatched tracks
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.time_since_update += 1
                track.hit_streak = 0
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]
        
        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks
            if track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
        ]
        
        return confirmed_tracks


class AppleTracker:
    """
    Apple Multi-Object Tracker using DeepSORT or Simple Tracker
    
    This class provides functionality for tracking detected apples across
    video frames to enable accurate counting and yield estimation.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        feature_dim: int = 128,
        matching_threshold: float = 0.7,
        use_deepsort: bool = True
    ):
        """
        Initialize Apple Tracker
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum hits required before track is confirmed
            iou_threshold: IoU threshold for data association
            feature_dim: Feature dimension for appearance embedding
            matching_threshold: Threshold for appearance matching
            use_deepsort: Whether to use DeepSORT (if available)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_dim = feature_dim
        self.matching_threshold = matching_threshold
        
        # Initialize tracker
        if use_deepsort and DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=min_hits,
                max_iou_distance=1 - iou_threshold,
                max_cosine_distance=matching_threshold,
                nn_budget=100
            )
            self.tracker_type = "DeepSORT"
            logger.info("Using DeepSORT tracker")
        else:
            self.tracker = SimpleTracker(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold
            )
            self.tracker_type = "Simple"
            logger.info("Using Simple tracker")
        
        # Tracking statistics
        self.total_unique_apples = 0
        self.active_tracks = 0
        self.track_history = defaultdict(list)
        self.apple_counts_per_frame = []
        
        # Performance metrics
        self.tracking_times = []
        
    def update(
        self, 
        detections: List[Dict],
        frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries from AppleDetector
            frame: Current frame image (required for DeepSORT)
            
        Returns:
            List of track dictionaries with updated information
        """
        start_time = time.time()
        
        if self.tracker_type == "DeepSORT":
            tracks = self._update_deepsort(detections, frame)
        else:
            tracks = self._update_simple(detections)
        
        # Update statistics
        self.active_tracks = len(tracks)
        current_ids = {track['track_id'] for track in tracks}
        
        if current_ids:
            max_id = max(current_ids)
            self.total_unique_apples = max(self.total_unique_apples, max_id + 1)
        
        self.apple_counts_per_frame.append(len(tracks))
        
        # Record track history
        for track in tracks:
            track_id = track['track_id']
            self.track_history[track_id].append({
                'frame': len(self.apple_counts_per_frame) - 1,
                'bbox': track['bbox'],
                'confidence': track['confidence']
            })
        
        tracking_time = time.time() - start_time
        self.tracking_times.append(tracking_time)
        
        return tracks
    
    def _update_deepsort(
        self, 
        detections: List[Dict],
        frame: np.ndarray
    ) -> List[Dict]:
        """Update using DeepSORT tracker"""
        if frame is None:
            logger.warning("Frame required for DeepSORT tracking")
            return []
        
        # Convert detections to DeepSORT format
        deepsort_detections = []
        for det in detections:
            for box, conf in zip(det['boxes'], det['confidence_scores']):
                # Convert to (left, top, width, height) format
                left = box['x1']
                top = box['y1']
                width = box['width']
                height = box['height']
                
                deepsort_detections.append([left, top, width, height, conf])
        
        # Update tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        # Convert tracks to our format
        formatted_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bbox = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format
            
            track_dict = {
                'track_id': track.track_id,
                'bbox': {
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3]),
                    'center_x': float((bbox[0] + bbox[2]) / 2),
                    'center_y': float((bbox[1] + bbox[3]) / 2),
                    'width': float(bbox[2] - bbox[0]),
                    'height': float(bbox[3] - bbox[1])
                },
                'confidence': float(track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5),
                'class_id': 0,  # Apple class
                'age': track.age if hasattr(track, 'age') else 0
            }
            formatted_tracks.append(track_dict)
        
        return formatted_tracks
    
    def _update_simple(self, detections: List[Dict]) -> List[Dict]:
        """Update using Simple tracker"""
        # Convert detections to simple format
        simple_detections = []
        for det in detections:
            for box, conf in zip(det['boxes'], det['confidence_scores']):
                detection = Detection(
                    bbox=(box['x1'], box['y1'], box['x2'], box['y2']),
                    confidence=conf,
                    class_id=0
                )
                simple_detections.append(detection)
        
        # Update tracker
        tracks = self.tracker.update(simple_detections)
        
        # Convert tracks to our format
        formatted_tracks = []
        for track in tracks:
            track_dict = {
                'track_id': track.track_id,
                'bbox': {
                    'x1': track.bbox[0],
                    'y1': track.bbox[1],
                    'x2': track.bbox[2],
                    'y2': track.bbox[3],
                    'center_x': (track.bbox[0] + track.bbox[2]) / 2,
                    'center_y': (track.bbox[1] + track.bbox[3]) / 2,
                    'width': track.bbox[2] - track.bbox[0],
                    'height': track.bbox[3] - track.bbox[1]
                },
                'confidence': track.confidence,
                'class_id': track.class_id,
                'age': track.age
            }
            formatted_tracks.append(track_dict)
        
        return formatted_tracks
    
    def get_unique_apple_count(self) -> int:
        """
        Get total count of unique apples detected
        
        Returns:
            Total number of unique apples
        """
        return self.total_unique_apples
    
    def get_active_tracks_count(self) -> int:
        """
        Get number of currently active tracks
        
        Returns:
            Number of active tracks
        """
        return self.active_tracks
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """
        Get history for specific track
        
        Args:
            track_id: ID of track to retrieve
            
        Returns:
            List of track history entries
        """
        return self.track_history.get(track_id, [])
    
    def get_all_track_histories(self) -> Dict[int, List[Dict]]:
        """
        Get all track histories
        
        Returns:
            Dictionary mapping track IDs to their histories
        """
        return dict(self.track_history)
    
    def visualize_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        show_ids: bool = True,
        show_trails: bool = True,
        trail_length: int = 10
    ) -> np.ndarray:
        """
        Visualize tracks on frame
        
        Args:
            frame: Input frame
            tracks: List of current tracks
            show_ids: Whether to show track IDs
            show_trails: Whether to show track trails
            trail_length: Length of trails to show
            
        Returns:
            Frame with track visualizations
        """
        vis_frame = frame.copy()
        
        # Color palette for different tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (int(bbox['x1']), int(bbox['y1'])),
                (int(bbox['x2']), int(bbox['y2'])),
                color,
                2
            )
            
            # Draw track ID
            if show_ids:
                label = f"ID: {track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    vis_frame,
                    (int(bbox['x1']), int(bbox['y1']) - label_size[1] - 10),
                    (int(bbox['x1']) + label_size[0], int(bbox['y1'])),
                    color,
                    -1
                )
                cv2.putText(
                    vis_frame,
                    label,
                    (int(bbox['x1']), int(bbox['y1']) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # Draw trail
            if show_trails and track_id in self.track_history:
                history = self.track_history[track_id]
                if len(history) > 1:
                    trail_points = []
                    for i, hist_entry in enumerate(history[-trail_length:]):
                        center_x = int((hist_entry['bbox']['x1'] + hist_entry['bbox']['x2']) / 2)
                        center_y = int((hist_entry['bbox']['y1'] + hist_entry['bbox']['y2']) / 2)
                        trail_points.append((center_x, center_y))
                    
                    # Draw trail lines
                    for i in range(1, len(trail_points)):
                        alpha = i / len(trail_points)  # Fade effect
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(vis_frame, trail_points[i-1], trail_points[i], trail_color, 2)
        
        # Add statistics
        stats_text = f"Active: {len(tracks)}, Total Unique: {self.total_unique_apples}"
        cv2.putText(
            vis_frame,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return vis_frame
    
    def export_tracking_data(self, output_path: str):
        """
        Export tracking data to CSV file
        
        Args:
            output_path: Path to save CSV file
        """
        import pandas as pd
        
        # Prepare data for export
        data = []
        for track_id, history in self.track_history.items():
            for entry in history:
                data.append({
                    'track_id': track_id,
                    'frame': entry['frame'],
                    'x1': entry['bbox']['x1'],
                    'y1': entry['bbox']['y1'],
                    'x2': entry['bbox']['x2'],
                    'y2': entry['bbox']['y2'],
                    'center_x': (entry['bbox']['x1'] + entry['bbox']['x2']) / 2,
                    'center_y': (entry['bbox']['y1'] + entry['bbox']['y2']) / 2,
                    'width': entry['bbox']['x2'] - entry['bbox']['x1'],
                    'height': entry['bbox']['y2'] - entry['bbox']['y1'],
                    'confidence': entry['confidence']
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Tracking data exported to {output_path}")
    
    def get_performance_stats(self) -> Dict:
        """
        Get tracking performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.tracking_times:
            return {}
        
        return {
            'avg_tracking_time': np.mean(self.tracking_times),
            'min_tracking_time': np.min(self.tracking_times),
            'max_tracking_time': np.max(self.tracking_times),
            'total_frames_processed': len(self.tracking_times),
            'fps': 1.0 / np.mean(self.tracking_times) if self.tracking_times else 0,
            'total_unique_apples': self.total_unique_apples,
            'avg_apples_per_frame': np.mean(self.apple_counts_per_frame) if self.apple_counts_per_frame else 0
        }
    
    def reset(self):
        """Reset tracker state"""
        if self.tracker_type == "DeepSORT":
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                max_iou_distance=1 - self.iou_threshold,
                max_cosine_distance=self.matching_threshold,
                nn_budget=100
            )
        else:
            self.tracker = SimpleTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )
        
        self.total_unique_apples = 0
        self.active_tracks = 0
        self.track_history.clear()
        self.apple_counts_per_frame.clear()
        self.tracking_times.clear()
        
        logger.info("Tracker reset completed")


if __name__ == "__main__":
    # Example usage
    tracker = AppleTracker(max_age=30, min_hits=3)
    
    # Example tracking update
    # detections = [...]  # From AppleDetector
    # frame = cv2.imread('frame.jpg')
    # tracks = tracker.update(detections, frame)
    # print(f"Active tracks: {len(tracks)}")
    # print(f"Total unique apples: {tracker.get_unique_apple_count()}")