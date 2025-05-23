# Integrating MLGestureRecognizer into Main Application Logic (`src/main.py`)

This document outlines the conceptual changes required to integrate the `MLGestureRecognizer` (from `src/modules/ml_gesture_recognizer.py`) into the main application loop in `src/main.py`. The goal is to allow switching between the `BasicGestureRecognizer` and the more advanced (though currently conceptual) `MLGestureRecognizer`.

## Core Changes to `src/main.py`

### 1. Imports:
Add the new recognizer to imports:
```python
from modules.ml_gesture_recognizer import MLGestureRecognizer
```

### 2. Recognizer Initialization:
Modify the initialization phase to allow selection of the recognizer, perhaps via a command-line argument or a configuration flag.

```python
# --- Inside main_loop() ---

# Configuration (example)
USE_ML_RECOGNIZER = True # This could come from args or a config file

# Initialize components
try:
    webcam = Webcam(camera_index=0)
    if USE_ML_RECOGNIZER:
        # Conceptual: Path to a trained model file would be needed here
        # e.g., model_file = "models/trained_gesture_model.h5" 
        # For now, we can pass the dummy path used in MLGestureRecognizer's tests
        model_file = "path/to/dummy_model.h5" 
        gesture_recognizer = MLGestureRecognizer(model_path=model_file)
        if gesture_recognizer.model is None and model_file is not None:
             print(f"WARNING: ML model at '{model_file}' could not be loaded. ML features may not work.")
        print("INFO: Using MLGestureRecognizer.")
    else:
        gesture_recognizer = BasicGestureRecognizer()
        print("INFO: Using BasicGestureRecognizer.")
    
    sign_db = SignDatabase() 
    output_handler = OutputHandler()

except Exception as e:
    print(f"ERROR: Initialization failed - {e}")
    return
```

### 3. Frame Processing and Prediction Loop:
The main loop structure would remain similar, but the methods called on `gesture_recognizer` would now be those of `MLGestureRecognizer` if it's selected.

```python
# --- Inside the while loop of main_loop() ---

frame = webcam.get_frame()
if frame is None:
    # ... (error handling) ...
    break 

if USE_ML_RECOGNIZER:
    # 1. Preprocess frame for ML model
    # Frame is currently "Simulated_Frame_Data" from our simulated webcam
    processed_ml_frame = gesture_recognizer.preprocess_frame(frame) 
    
    # 2. Predict using ML model
    # This returns (gesture_key, confidence_score)
    gesture_key, confidence = gesture_recognizer.predict(processed_ml_frame) 
    
    if gesture_key and confidence > 0.5: # Example confidence threshold
        print(f"DEBUG: ML Prediction: Key='{gesture_key}', Confidence={confidence:.2f}")
        gesture_data = sign_db.get_gesture_data(gesture_key) # Use key to get text/audio
    else:
        gesture_data = None
        if gesture_key: # Had a key but low confidence
             print(f"DEBUG: ML Prediction '{gesture_key}' below confidence threshold ({confidence:.2f}).")

else: # Using BasicGestureRecognizer
    detected_outline = gesture_recognizer.detect_hand_outline(frame)
    if detected_outline:
        print(f"DEBUG: Basic Detected outline: {detected_outline}")
        gesture_data = gesture_recognizer.compare_gestures(
            detected_outline,
            sign_db.get_all_gestures()
        )
    else:
        gesture_data = None

# 4. Output (common for both recognizers)
if gesture_data:
    output_handler.display_text(gesture_data.get("text"))
    output_handler.play_audio(gesture_data.get("audio"))
else:
    if USE_ML_RECOGNIZER and gesture_key and confidence <= 0.5 and confidence > 0.0:
        output_handler.display_text(f"Low confidence for {gesture_key} ({confidence:.2f}).")
    elif USE_ML_RECOGNIZER and not gesture_key:
         output_handler.display_text("No gesture recognized by ML model.")
    else: # Basic recognizer path
        output_handler.display_text("Unknown gesture or no match (basic).")

```

## Considerations for Actual Implementation:

*   **Model Availability:** A trained `.h5` (or equivalent) model file would need to be available at the specified `model_path`.
*   **Input Consistency:** The `MLGestureRecognizer.preprocess_frame` method would need to correctly transform the output of `webcam.get_frame()` (which is currently a string "Simulated_Frame_Data", but in reality an OpenCV image frame) into the exact format expected by the actual ML model (e.g., NumPy array of specific shape, type, and normalization).
*   **Output Mapping:** The `gesture_key` returned by `MLGestureRecognizer.predict` must correspond to keys present in `data/sign_db.json` for the application to retrieve text and audio details. The ML model's output classes would need to be mapped to these keys.
*   **Performance:** Real ML model inference can be computationally intensive. The main loop might need adjustments for real-time performance, potentially involving threading or asynchronous processing if inference time is high.
*   **Error Handling:** Robust error handling for model loading failures, preprocessing issues, and prediction errors would be essential.

These changes would allow the `main.py` to leverage the (conceptual) ML-based recognition while retaining the basic simulated recognizer as a fallback or for comparison.
