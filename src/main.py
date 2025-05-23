# src/main.py
from modules.webcam import Webcam
from modules.gesture_recognition import BasicGestureRecognizer
from modules.database import SignDatabase
from modules.output import OutputHandler
# import time # Potentially for adding delays or managing loop frequency

def main_loop():
    print("INFO: Starting Sign Language Recognition Application (Simulated)")

    # Initialize components
    try:
        webcam = Webcam(camera_index=0)
        gesture_recognizer = BasicGestureRecognizer()
        # The SignDatabase by default loads from 'data/sign_db.json'
        # This file will be created with actual data in a later plan step.
        # For now, it might be empty or non-existent (database module handles this)
        sign_db = SignDatabase() 
        output_handler = OutputHandler()
    except Exception as e:
        print(f"ERROR: Initialization failed - {e}")
        return

    if not webcam.start_capture():
        print("ERROR: Failed to start webcam. Exiting application.")
        return

    print("INFO: Webcam started. Entering main loop (simulated capture and processing)...")
    print("INFO: Press Ctrl+C (or close window if UI existed) to exit.")

    try:
        # Counter for simulated frames to eventually stop the loop in simulation
        simulated_frame_count = 0
        max_simulated_frames = 5 # Stop after a few simulated frames

        while simulated_frame_count < max_simulated_frames:
            print(f"\n--- Frame {simulated_frame_count + 1} ---")
            frame = webcam.get_frame()

            if frame is None:
                print("WARNING: Could not get frame from webcam. Skipping this iteration.")
                # In a real app, might attempt to re-initialize webcam or exit
                # For simulation, we might just stop if frames stop coming
                break 
            
            # 1. Detect hand outline (simulated)
            # In reality, frame would be an image (e.g., numpy array)
            detected_outline = gesture_recognizer.detect_hand_outline(frame)

            if detected_outline:
                print(f"DEBUG: Detected outline: {detected_outline}")
                
                # 2. Compare with database (simulated)
                # The gesture_recognizer.compare_gestures expects the raw db dictionary
                gesture_data = gesture_recognizer.compare_gestures(
                    detected_outline, 
                    sign_db.get_all_gestures() # Pass the whole gesture map
                )
                # Alternative: query the database directly if comparison logic was simpler
                # gesture_data = sign_db.get_gesture_data(detected_outline)


                if gesture_data:
                    # 3. Output text and audio (simulated)
                    output_handler.display_text(gesture_data.get("text"))
                    output_handler.play_audio(gesture_data.get("audio"))
                else:
                    output_handler.display_text("Unknown gesture or no match.")
                    # print("INFO: No matching gesture in the database for the detected outline.")
            else:
                # print("INFO: No hand outline detected in the current frame.")
                output_handler.display_text("No hand detected.")

            simulated_frame_count += 1
            # time.sleep(1) # Simulate delay between frames if running too fast

    except KeyboardInterrupt:
        print("\nINFO: Keyboard interrupt received. Exiting application...")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in the main loop: {e}")
    finally:
        print("INFO: Releasing webcam and cleaning up...")
        webcam.release()
        print("INFO: Application finished.")

if __name__ == "__main__":
    # Note: For this main.py to run meaningfully, data/sign_db.json should exist
    # and contain entries that 'simulated_hand_outline_features' can match.
    # This will be handled in the next step "Create Placeholder Data".
    # If you run this now, it will likely report "Unknown gesture" as the DB is empty.
    
    # To make this testable now, let's quickly ensure data/sign_db.json exists
    # and has at least one entry that the simulated gesture recognizer can find.
    # This is a temporary measure for immediate testing of main_loop.
    # The proper creation of this file is part of the next plan step.
    
    import os
    import json
    temp_db_path_for_main = "data/sign_db.json"
    temp_db_dir_for_main = os.path.dirname(temp_db_path_for_main)
    
    # Ensure directory exists
    if not os.path.exists(temp_db_dir_for_main):
        os.makedirs(temp_db_dir_for_main, exist_ok=True)
        print(f"INFO: Created directory {temp_db_dir_for_main} for main.py test.")

    # Create a minimal DB for the simulation to work
    # This matches what BasicGestureRecognizer.detect_hand_outline returns
    # and what BasicGestureRecognizer.compare_gestures expects as a key.
    minimal_db_content = {
        "simulated_hand_outline_features": {
            "text": "Simulated Sign Detected",
            "audio": "data/audio/simulated_sign.mp3" # This audio file won't exist yet
        }
    }
    # Only write if the file doesn't exist or is empty, to avoid overwriting if previously set up
    create_temp_db = True
    if os.path.exists(temp_db_path_for_main):
        try:
            with open(temp_db_path_for_main, 'r') as f:
                existing_data = json.load(f)
            if existing_data: # If file exists and has content, don't overwrite for this test
                create_temp_db = False
                print(f"INFO: {temp_db_path_for_main} already exists and has content. Main loop will use it.")
        except (json.JSONDecodeError, FileNotFoundError): # If corrupted or somehow gone after check
             create_temp_db = True # Proceed to create/overwrite

    if create_temp_db:
        print(f"INFO: Creating/Overwriting minimal {temp_db_path_for_main} for main.py test.")
        with open(temp_db_path_for_main, 'w') as f:
            json.dump(minimal_db_content, f, indent=4)
        # Also ensure the dummy audio directory exists, as referenced by the minimal DB
        dummy_audio_referenced_dir = "data/audio/"
        if not os.path.exists(dummy_audio_referenced_dir):
            os.makedirs(dummy_audio_referenced_dir, exist_ok=True)
            print(f"INFO: Created directory {dummy_audio_referenced_dir} for main.py audio reference.")
        # And a dummy audio file
        dummy_audio_file_path = os.path.join(dummy_audio_referenced_dir, "simulated_sign.mp3")
        if not os.path.exists(dummy_audio_file_path):
            with open(dummy_audio_file_path, 'w') as fda:
                fda.write("dummy audio") # create empty file
            print(f"INFO: Created dummy audio file {dummy_audio_file_path} for main.py test.")


    main_loop()
