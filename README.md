# Sign Language Gesture Recognition (Simulated)

This project is a Python-based application that simulates the core components of a sign language gesture recognition system. It includes modules for (simulated) webcam access, basic gesture detection, a gesture database, and outputting results as text and (simulated) audio. It also outlines the conceptual integration of a Machine Learning based gesture recognizer.

**Note:** This is primarily a structural and conceptual framework. The gesture recognition is currently rule-based and simulated, and the Machine Learning components are placeholders to illustrate how they would integrate into the system.

## Project Structure

*   `src/`: Contains the main source code.
    *   `main.py`: The main application script to run the simulation.
    *   `modules/`: Core functional modules.
        *   `webcam.py`: Simulated webcam access.
        *   `gesture_recognition.py`: Basic (simulated) hand outline detection and comparison.
        *   `database.py`: Handles loading and querying the sign gesture database.
        *   `output.py`: Handles text display and (simulated) audio playback.
        *   `ml_gesture_recognizer.py`: Conceptual placeholder for an ML-based recognizer.
    *   `utils/`: Utility functions (currently `helpers.py` is empty).
*   `data/`: Contains data files.
    *   `sign_db.json`: A simple JSON database mapping simulated gesture features to text and audio paths.
    *   `audio/`: Contains dummy .mp3 files for simulated audio playback.
*   `tests/`: Contains unit tests for the modules.
    *   `test_*.py`: Individual test files for each module.
*   `docs/`: Contains documentation files.
    *   `ml_data_strategy.md`: Conceptual strategy for ML data collection & preprocessing.
    *   `ml_integration_main.md`: Conceptual guide for integrating the ML recognizer into `main.py`.

## Features (Simulated)

*   **Webcam Input:** Simulates capturing frames from a webcam.
*   **Gesture Detection:** Implements a very basic, rule-based "hand outline" detection.
*   **Gesture Database:** Uses a JSON file to store predefined gesture representations and their corresponding text/audio.
*   **Recognition:** Compares detected "gestures" against the database.
*   **Output:** Displays recognized text to the console and simulates audio playback by printing the path to an audio file.
*   **ML Placeholder:** Includes a conceptual `MLGestureRecognizer` class to show where a real ML model would fit.
*   **Unit Tests:** Basic unit tests for each module to verify functionality.

## Setup and Installation

1.  **Python Version:** Python 3.7+ is recommended.
2.  **Clone the Repository (Example):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
3.  **Dependencies:**
    This project primarily uses standard Python libraries. For a full version with actual hardware interaction and ML, you would need libraries like:
    *   `opencv-python`: For webcam access and image processing.
    *   `numpy`: For numerical operations, especially with image data.
    *   A deep learning framework like `tensorflow` or `pytorch` for the ML model.
    *   Audio playback libraries like `pygame` or `playsound`.
    *   Text-to-speech library like `gTTS` (if generating audio from text).

    A conceptual `requirements.txt` might look like this:
    ```
    # Conceptual requirements for a full implementation
    # opencv-python>=4.5
    # numpy>=1.20
    # tensorflow>=2.5 # or torch>=1.8
    # pygame>=2.0 # for audio playback
    # gTTS>=2.2 # for text-to-speech
    ```
    For the current simulation, no external libraries beyond standard Python are strictly required to run `src/main.py` as the hardware-dependent parts are simulated.

## How to Run

1.  **Navigate to the Source Directory:**
    ```bash
    cd path/to/this/project/
    ```
2.  **Run the Main Application:**
    ```bash
    python src/main.py
    ```
    This will start the simulated application. It will print messages to the console indicating simulated frame capture, gesture detection, and any recognized gestures based on the `data/sign_db.json` file. The simulation runs for a few frames and then exits.

## How to Run Tests

1.  **Navigate to the Project Root Directory.**
2.  **Run Unittests:**
    ```bash
    python -m unittest discover -s tests
    ```
    This will discover and run all tests in the `tests/` directory.

## Conceptual ML Integration

The project includes a conceptual framework for integrating a Machine Learning model:
*   `src/modules/ml_gesture_recognizer.py`: A placeholder class for an ML recognizer.
*   `docs/ml_data_strategy.md`: Outlines strategies for data collection and preprocessing.
*   `docs/ml_integration_main.md`: Describes how the ML recognizer could be integrated into `main.py`.

Actually implementing and training an ML model is a significant task beyond the current simulation's scope and would require a substantial dataset and ML expertise.

## Future Enhancements (Conceptual)

*   Implement actual webcam capture using OpenCV.
*   Develop and integrate a real hand tracking and gesture recognition ML model.
*   Implement actual audio playback.
*   Create a graphical user interface (GUI) using Tkinter, PyQt, or a web framework.
*   Expand the gesture database.
*   Add configuration options (e.g., select camera, choose recognizer type).
```
