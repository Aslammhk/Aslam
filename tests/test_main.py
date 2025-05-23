# tests/test_main.py
import unittest
from unittest.mock import patch, MagicMock
# Import main_loop if it's directly callable, or structure to run it
from src.main import main_loop 

class TestMain(unittest.TestCase):

    @patch('src.main.Webcam')
    @patch('src.main.BasicGestureRecognizer')
    @patch('src.main.SignDatabase')
    @patch('src.main.OutputHandler')
    @patch('builtins.print') # To suppress print statements from main_loop during test
    def test_main_loop_runs_without_crashing_simulated(self, mock_print, MockOutputHandler, MockSignDatabase, MockGestureRecognizer, MockWebcam):
        # Configure mocks for a very short run
        
        # Mock Webcam
        mock_webcam_instance = MockWebcam.return_value
        mock_webcam_instance.start_capture.return_value = True
        # Simulate a couple of frames then None to stop loop
        mock_webcam_instance.get_frame.side_effect = ["Simulated_Frame_Data", "Simulated_Frame_Data_2", None]
        
        # Mock BasicGestureRecognizer
        mock_recognizer_instance = MockGestureRecognizer.return_value
        # Simulate detecting a gesture for the first frame, then not for the second
        mock_recognizer_instance.detect_hand_outline.side_effect = ["simulated_hand_outline_features", None]
        # Simulate a gesture match for the first detected outline
        mock_recognizer_instance.compare_gestures.return_value = {"text": "Test", "audio": "test.mp3"}

        # Mock SignDatabase
        mock_db_instance = MockSignDatabase.return_value
        mock_db_instance.get_all_gestures.return_value = {
            "simulated_hand_outline_features": {"text": "Test", "audio": "test.mp3"}
        } # Provide a minimal DB for the comparison

        # Mock OutputHandler
        mock_output_instance = MockOutputHandler.return_value
        # mock_output_instance.display_text = MagicMock()
        # mock_output_instance.play_audio = MagicMock()

        try:
            # Limit max_simulated_frames for testing if main_loop uses it
            # This is tricky if it's hardcoded in main_loop.
            # The get_frame side_effect returning None is the primary stop condition here.
            # We also need to consider the 'simulated_frame_count' and 'max_simulated_frames'
            # in main_loop. The test will run for at most 'max_simulated_frames' if not
            # stopped by get_frame returning None earlier.
            # For this test, main_loop's internal frame limit (max_simulated_frames = 5)
            # is more than the 3 side_effects for get_frame, so get_frame returning None
            # will be the actual stop condition.
            main_loop() 
        except Exception as e:
            self.fail(f"main_loop() raised an exception unexpectedly: {e}")

        # Assertions: Check if key methods were called
        mock_webcam_instance.start_capture.assert_called_once()
        self.assertTrue(mock_webcam_instance.get_frame.call_count >= 2, 
                        f"get_frame call count was {mock_webcam_instance.get_frame.call_count}") # Called for two frames + None
        
        # Check calls to detect_hand_outline
        # First call with "Simulated_Frame_Data", second with "Simulated_Frame_Data_2"
        mock_recognizer_instance.detect_hand_outline.assert_any_call("Simulated_Frame_Data")
        mock_recognizer_instance.detect_hand_outline.assert_any_call("Simulated_Frame_Data_2")
        
        # compare_gestures is called only when detect_hand_outline returns a non-None value
        # In this test, it's called once with "simulated_hand_outline_features"
        mock_recognizer_instance.compare_gestures.assert_called_once_with(
            "simulated_hand_outline_features", 
            mock_db_instance.get_all_gestures()
        )
        
        # Output handler calls
        # display_text is called with "Test" for the matched gesture,
        # and "No hand detected." or "Unknown gesture..." for the second frame and subsequent.
        mock_output_instance.display_text.assert_any_call("Test") 
        # play_audio is called with "test.mp3" for the matched gesture
        mock_output_instance.play_audio.assert_any_call("test.mp3")
        
        mock_webcam_instance.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()
