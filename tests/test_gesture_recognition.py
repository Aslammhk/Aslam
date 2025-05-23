# tests/test_gesture_recognition.py
import unittest
from src.modules.gesture_recognition import BasicGestureRecognizer

class TestBasicGestureRecognizer(unittest.TestCase):

    def setUp(self):
        self.recognizer = BasicGestureRecognizer()
        self.sample_db_gestures = {
            "simulated_hand_outline_features": {"text": "Generic Gesture", "audio": "audio/generic.mp3"},
            "specific_gesture_A": {"text": "Gesture A", "audio": "audio/a.mp3"},
            "specific_gesture_B": {"text": "Gesture B", "audio": "audio/b.mp3"}
        }

    def test_initialization(self):
        self.assertIsNotNone(self.recognizer, "Recognizer should initialize.")

    def test_detect_hand_outline_simulated_valid_frame(self):
        """Test hand outline detection with simulated valid frame data."""
        # This input matches what the simulated webcam provides
        frame_data = "Simulated_Frame_Data" 
        outline = self.recognizer.detect_hand_outline(frame_data)
        self.assertEqual(outline, "simulated_hand_outline_features", 
                         "Should return simulated outline for valid frame data.")

    def test_detect_hand_outline_simulated_none_frame(self):
        """Test hand outline detection with None as frame data."""
        frame_data = None
        outline = self.recognizer.detect_hand_outline(frame_data)
        self.assertIsNone(outline, "Should return None for None frame data.")

    def test_detect_hand_outline_simulated_unknown_frame(self):
        """Test hand outline detection with unrecognized frame data."""
        frame_data = "Some_Other_Frame_Data_Not_From_Webcam_Sim"
        outline = self.recognizer.detect_hand_outline(frame_data)
        self.assertIsNone(outline, 
                          "Should return None for frame data not matching 'Simulated_Frame_Data'.")

    def test_compare_gestures_direct_match(self):
        """Test gesture comparison with a direct match."""
        detected_features = "specific_gesture_A"
        match = self.recognizer.compare_gestures(detected_features, self.sample_db_gestures)
        self.assertIsNotNone(match)
        self.assertEqual(match["text"], "Gesture A")

    def test_compare_gestures_simulated_generic_match(self):
        """Test gesture comparison with the generic simulated outline features."""
        detected_features = "simulated_hand_outline_features" # From detect_hand_outline
        match = self.recognizer.compare_gestures(detected_features, self.sample_db_gestures)
        self.assertIsNotNone(match)
        self.assertEqual(match["text"], "Generic Gesture")

    def test_compare_gestures_no_match(self):
        """Test gesture comparison with no matching features in the database."""
        detected_features = "nonexistent_features"
        match = self.recognizer.compare_gestures(detected_features, self.sample_db_gestures)
        self.assertIsNone(match, "Should return None if no gesture matches.")

    def test_compare_gestures_empty_features(self):
        """Test gesture comparison with empty detected features."""
        detected_features = ""
        match = self.recognizer.compare_gestures(detected_features, self.sample_db_gestures)
        self.assertIsNone(match, "Should return None for empty detected features.")

    def test_compare_gestures_empty_database(self):
        """Test gesture comparison with an empty database."""
        detected_features = "simulated_hand_outline_features"
        match = self.recognizer.compare_gestures(detected_features, {})
        self.assertIsNone(match, "Should return None for an empty database.")

if __name__ == '__main__':
    unittest.main()
