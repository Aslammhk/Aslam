# src/modules/database.py
import json
import os

class SignDatabase:
    def __init__(self, db_path="data/sign_db.json"):
        """
        Initialize the SignDatabase.

        Args:
            db_path (str): Path to the JSON database file.
        """
        self.db_path = db_path
        self.gestures = {}
        self._load_database()

    def _load_database(self):
        """
        Loads the sign gesture database from the JSON file.
        The JSON file should contain a dictionary where keys are gesture
        feature representations (strings) and values are dictionaries
        with "text" and "audio" (path to audio file) keys.
        Example:
        {
            "gesture_feature_1": {"text": "Hello", "audio": "audio/hello.mp3"},
            "gesture_feature_2": {"text": "Thank You", "audio": "audio/thankyou.mp3"}
        }
        """
        try:
            # Ensure the data directory exists, though file itself might not yet
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            if not os.path.exists(self.db_path):
                print(f"WARNING: Database file not found at {self.db_path}. Creating an empty one for now.")
                # Create an empty JSON file if it doesn't exist to prevent load error later
                # The actual data will be added in a subsequent plan step.
                with open(self.db_path, 'w') as f:
                    json.dump({}, f)
                self.gestures = {}
                return

            with open(self.db_path, 'r') as f:
                self.gestures = json.load(f)
            print(f"INFO: Sign database loaded successfully from {self.db_path}.")
            if not self.gestures:
                print(f"WARNING: The database at {self.db_path} is empty.")

        except FileNotFoundError:
            # This case should be handled by the os.path.exists check and creation now
            print(f"ERROR: Database file not found at {self.db_path}. An empty database will be used.")
            self.gestures = {}
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {self.db_path}. Corrupted file? An empty database will be used.")
            self.gestures = {}
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading the database: {e}")
            self.gestures = {}

    def get_gesture_data(self, gesture_features_key):
        """
        Retrieves gesture data (text, audio path) for a given feature key.

        Args:
            gesture_features_key (str): The key representing the gesture features.

        Returns:
            dict: A dictionary with "text" and "audio" if the key is found, otherwise None.
        """
        return self.gestures.get(gesture_features_key)

    def get_all_gestures(self):
        """
        Returns all loaded gestures.

        Returns:
            dict: The entire gestures dictionary.
        """
        return self.gestures

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy database file for testing this module directly
    # In the actual plan, 'data/sign_db.json' will be created by a later step.
    # For now, we ensure the directory exists and create a temporary DB.
    
    temp_db_path = "temp_sign_db_for_test.json"
    temp_db_data_dir = os.path.dirname(temp_db_path)
    if temp_db_data_dir and not os.path.exists(temp_db_data_dir): # Ensure dir exists if path includes one
        os.makedirs(temp_db_data_dir, exist_ok=True)

    sample_data = {
        "simulated_hand_outline_features": {"text": "Hello World", "audio": "audio/helloworld.mp3"},
        "gesture_test_01": {"text": "Test One", "audio": "audio/testone.mp3"}
    }
    with open(temp_db_path, 'w') as f:
        json.dump(sample_data, f)

    print(f"--- Testing with temporary database: {temp_db_path} ---")
    db = SignDatabase(db_path=temp_db_path)
    print(f"Loaded gestures: {db.get_all_gestures()}")

    gesture_key = "simulated_hand_outline_features"
    data = db.get_gesture_data(gesture_key)
    if data:
        print(f"Data for '{gesture_key}': Text='{data['text']}', Audio='{data['audio']}'")
    else:
        print(f"No data found for '{gesture_key}'.")

    unknown_key = "unknown_gesture"
    data_unknown = db.get_gesture_data(unknown_key)
    if data_unknown:
        print(f"Data for '{unknown_key}': {data_unknown}")
    else:
        print(f"No data found for '{unknown_key}' (as expected).")
        
    # Test with a non-existent DB path (it should create an empty one)
    # Ensure it doesn't crash and uses an empty gesture dict
    print("--- Testing with non-existent DB (should create empty one) ---")
    non_existent_db_path = "non_existent_db.json"
    if os.path.exists(non_existent_db_path):
        os.remove(non_existent_db_path) # ensure it's gone
        
    db_non_existent = SignDatabase(db_path=non_existent_db_path)
    print(f"Gestures from non-existent DB: {db_non_existent.get_all_gestures()}")
    if os.path.exists(non_existent_db_path): # Check if it was created
        print(f"INFO: '{non_existent_db_path}' was created as expected.")
        os.remove(non_existent_db_path) # Clean up
    else:
        print(f"ERROR: '{non_existent_db_path}' was not created.")


    # Clean up the temporary database file
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
