# tests/test_output.py
import unittest
from unittest.mock import patch
import io
import os
from src.modules.output import OutputHandler

class TestOutputHandler(unittest.TestCase):

    def setUp(self):
        self.output_handler = OutputHandler()
        # Create a dummy audio file for tests that need one
        self.test_audio_dir = "temp_test_audio"
        os.makedirs(self.test_audio_dir, exist_ok=True)
        self.dummy_audio_file = os.path.join(self.test_audio_dir, "test.mp3")
        with open(self.dummy_audio_file, "w") as f:
            f.write("dummy content")

    def tearDown(self):
        # Clean up the dummy audio file and directory
        if os.path.exists(self.dummy_audio_file):
            os.remove(self.dummy_audio_file)
        if os.path.exists(self.test_audio_dir):
            os.rmdir(self.test_audio_dir)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_display_text_valid(self, mock_stdout):
        self.output_handler.display_text("Test message")
        self.assertIn("Gesture Recognized: Test message", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_display_text_none(self, mock_stdout):
        self.output_handler.display_text(None)
        self.assertIn("INFO: No text to display.", mock_stdout.getvalue())

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_play_audio_simulated_success(self, mock_stdout):
        self.output_handler.play_audio(self.dummy_audio_file)
        output = mock_stdout.getvalue()
        self.assertIn(f"Attempting to play audio from '{self.dummy_audio_file}' (simulated).", output)
        self.assertIn(f"SUCCESS: Audio file '{self.dummy_audio_file}' exists and would be played (simulated).", output)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_play_audio_file_not_found(self, mock_stdout):
        non_existent_file = "non_existent.mp3"
        self.output_handler.play_audio(non_existent_file)
        output = mock_stdout.getvalue()
        self.assertIn(f"Attempting to play audio from '{non_existent_file}' (simulated).", output)
        self.assertIn(f"ERROR: Simulated audio playback failed: Audio file not found at '{non_existent_file}'.", output)
        
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_play_audio_none_path(self, mock_stdout):
        self.output_handler.play_audio(None)
        self.assertIn("WARNING: No audio file path provided.", mock_stdout.getvalue())

if __name__ == '__main__':
    unittest.main()
