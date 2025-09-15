"""
Unit tests for Apple Tracker module
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models.apple_tracker import AppleTracker, SimpleTracker, Detection, Track
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestSimpleTracker(unittest.TestCase):
    """Test cases for SimpleTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = SimpleTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.max_age, 5)
        self.assertEqual(self.tracker.min_hits, 2)
        self.assertEqual(self.tracker.iou_threshold, 0.3)
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.track_count, 0)
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        
        iou = self.tracker._calculate_iou(box1, box2)
        
        self.assertIsInstance(iou, float)
        self.assertGreaterEqual(iou, 0)
        self.assertLessEqual(iou, 1)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap"""
        box1 = (10, 10, 50, 50)
        box2 = (60, 60, 100, 100)
        
        iou = self.tracker._calculate_iou(box1, box2)
        
        self.assertEqual(iou, 0.0)
    
    def test_update_with_detections(self):
        """Test tracker update with detections"""
        detections = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.8, class_id=0),
            Detection(bbox=(60, 60, 100, 100), confidence=0.9, class_id=0)
        ]
        
        tracks = self.tracker.update(detections)
        
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertEqual(self.tracker.track_count, 2)
    
    def test_update_empty_detections(self):
        """Test tracker update with empty detections"""
        tracks = self.tracker.update([])
        
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 0)
    
    def test_track_aging(self):
        """Test track aging mechanism"""
        # Add initial detection
        detections = [Detection(bbox=(10, 10, 50, 50), confidence=0.8, class_id=0)]
        self.tracker.update(detections)
        
        # Update with no detections multiple times
        for _ in range(3):
            self.tracker.update([])
        
        # Check that unmatched tracks age
        self.assertEqual(len(self.tracker.tracks), 1)
        track = self.tracker.tracks[0]
        self.assertEqual(track.time_since_update, 3)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestAppleTracker(unittest.TestCase):
    """Test cases for AppleTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = AppleTracker(use_deepsort=False)  # Use simple tracker for testing
        self.test_detections = [
            {
                'boxes': [
                    {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50},
                    {'x1': 60, 'y1': 60, 'x2': 100, 'y2': 100}
                ],
                'confidence_scores': [0.8, 0.9],
                'total_apples': 2
            }
        ]
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.tracker_type, "Simple")
        self.assertEqual(self.tracker.total_unique_apples, 0)
        self.assertEqual(self.tracker.active_tracks, 0)
    
    def test_update_with_detections(self):
        """Test tracker update with detections"""
        tracks = self.tracker.update(self.test_detections, self.test_frame)
        
        self.assertIsInstance(tracks, list)
        for track in tracks:
            self.assertIn('track_id', track)
            self.assertIn('bbox', track)
            self.assertIn('confidence', track)
    
    def test_update_empty_detections(self):
        """Test tracker update with empty detections"""
        empty_detections = [{'boxes': [], 'confidence_scores': [], 'total_apples': 0}]
        tracks = self.tracker.update(empty_detections, self.test_frame)
        
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 0)
    
    def test_get_unique_apple_count(self):
        """Test unique apple count"""
        self.tracker.update(self.test_detections, self.test_frame)
        count = self.tracker.get_unique_apple_count()
        
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)
    
    def test_get_active_tracks_count(self):
        """Test active tracks count"""
        self.tracker.update(self.test_detections, self.test_frame)
        count = self.tracker.get_active_tracks_count()
        
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)
    
    def test_track_history(self):
        """Test track history functionality"""
        tracks = self.tracker.update(self.test_detections, self.test_frame)
        
        if tracks:
            track_id = tracks[0]['track_id']
            history = self.tracker.get_track_history(track_id)
            
            self.assertIsInstance(history, list)
            if history:
                self.assertIn('frame', history[0])
                self.assertIn('bbox', history[0])
    
    def test_get_all_track_histories(self):
        """Test getting all track histories"""
        self.tracker.update(self.test_detections, self.test_frame)
        all_histories = self.tracker.get_all_track_histories()
        
        self.assertIsInstance(all_histories, dict)
    
    def test_visualize_tracks(self):
        """Test track visualization"""
        tracks = self.tracker.update(self.test_detections, self.test_frame)
        
        if tracks:
            visualized = self.tracker.visualize_tracks(
                self.test_frame, tracks, show_ids=True, show_trails=False
            )
            
            self.assertEqual(visualized.shape, self.test_frame.shape)
            self.assertIsInstance(visualized, np.ndarray)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        self.tracker.update(self.test_detections, self.test_frame)
        stats = self.tracker.get_performance_stats()
        
        if stats:  # Only check if there are stats
            self.assertIn('avg_tracking_time', stats)
            self.assertIn('total_frames_processed', stats)
    
    def test_reset_tracker(self):
        """Test tracker reset"""
        self.tracker.update(self.test_detections, self.test_frame)
        
        # Reset tracker
        self.tracker.reset()
        
        self.assertEqual(self.tracker.total_unique_apples, 0)
        self.assertEqual(self.tracker.active_tracks, 0)
        self.assertEqual(len(self.tracker.track_history), 0)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestDataStructures(unittest.TestCase):
    """Test data structures used in tracking"""
    
    def test_detection_creation(self):
        """Test Detection dataclass"""
        detection = Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.8,
            class_id=0
        )
        
        self.assertEqual(detection.bbox, (10, 10, 50, 50))
        self.assertEqual(detection.confidence, 0.8)
        self.assertEqual(detection.class_id, 0)
    
    def test_track_creation(self):
        """Test Track dataclass"""
        track = Track(
            track_id=1,
            bbox=(10, 10, 50, 50),
            confidence=0.8,
            class_id=0,
            age=5
        )
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.bbox, (10, 10, 50, 50))
        self.assertEqual(track.confidence, 0.8)
        self.assertEqual(track.age, 5)


if __name__ == '__main__':
    unittest.main()