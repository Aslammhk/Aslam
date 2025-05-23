# tests/test_database.py
import unittest
import json
import os
from src.modules.database import SignDatabase

class TestSignDatabase(unittest.TestCase):

    def setUp(self):
        self.test_db_path = "test_sign_db.json"
        self.test_data_dir = os.path.dirname(self.test_db_path) # Might be empty if file in current dir
        if self.test_data_dir and not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir, exist_ok=True)

        self.sample_gestures = {
            "feature_A": {"text": "Sign A", "audio": "audio/a.mp3"},
            "feature_B": {"text": "Sign B", "audio": "audio/b.mp3"}
        }
        with open(self.test_db_path, 'w') as f:
            json.dump(self.sample_gestures, f)

        self.empty_db_path = "empty_test_db.json"
        with open(self.empty_db_path, 'w') as f:
            json.dump({}, f)
            
        self.corrupted_db_path = "corrupted_test_db.json"
        with open(self.corrupted_db_path, 'w') as f:
            f.write("this is not valid json")

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists(self.empty_db_path):
            os.remove(self.empty_db_path)
        if os.path.exists(self.corrupted_db_path):
            os.remove(self.corrupted_db_path)
        # Check if a default db was created by a test and remove it
        default_db_path_for_creation_test = "new_db_created.json"
        if os.path.exists(default_db_path_for_creation_test):
            os.remove(default_db_path_for_creation_test)


    def test_load_database_success(self):
        db = SignDatabase(db_path=self.test_db_path)
        self.assertEqual(db.get_all_gestures(), self.sample_gestures)

    def test_load_database_file_not_found_creates_empty(self):
        non_existent_path = "new_db_created.json" # Use a unique name for this test
        if os.path.exists(non_existent_path): # ensure it's clean before test
            os.remove(non_existent_path)
            
        db = SignDatabase(db_path=non_existent_path)
        self.assertEqual(db.get_all_gestures(), {})
        self.assertTrue(os.path.exists(non_existent_path), "Database file should be created if not found.")
        # Clean up after this specific test
        if os.path.exists(non_existent_path):
            os.remove(non_existent_path)


    def test_load_empty_database(self):
        db = SignDatabase(db_path=self.empty_db_path)
        self.assertEqual(db.get_all_gestures(), {})

    def test_load_corrupted_database(self):
        # Suppress expected error messages during this test
        import sys
        from io import StringIO
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        
        db = SignDatabase(db_path=self.corrupted_db_path)
        
        sys.stderr = original_stderr # Restore stderr
        self.assertEqual(db.get_all_gestures(), {}, "Gestures should be empty for corrupted DB.")

    def test_get_gesture_data_found(self):
        db = SignDatabase(db_path=self.test_db_path)
        data = db.get_gesture_data("feature_A")
        self.assertEqual(data, self.sample_gestures["feature_A"])

    def test_get_gesture_data_not_found(self):
        db = SignDatabase(db_path=self.test_db_path)
        data = db.get_gesture_data("feature_X") # Non-existent key
        self.assertIsNone(data)

    def test_get_gesture_data_empty_db(self):
        db = SignDatabase(db_path=self.empty_db_path)
        data = db.get_gesture_data("feature_A")
        self.assertIsNone(data)
        
    def test_default_db_path_creation(self):
        # Tests if SignDatabase() uses 'data/sign_db.json' and potentially creates it.
        default_path = "data/sign_db.json"
        # Ensure data directory exists as per class logic
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        if os.path.exists(default_path):
            os.remove(default_path) # Clean before test
            
        db = SignDatabase() # Call with default path
        self.assertTrue(os.path.exists(default_path), 
                        f"Default database file '{default_path}' should be created if not found.")
        self.assertEqual(db.get_all_gestures(),{}) # Should be empty
        # Clean up the default file if it was created by this test
        if os.path.exists(default_path):
             # Check if it was created by this test run, not pre-existing from other steps
             with open(default_path, 'r') as f:
                 content = json.load(f)
             if content == {}: # only remove if it's the empty one we just made
                 os.remove(default_path)


if __name__ == '__main__':
    unittest.main()
