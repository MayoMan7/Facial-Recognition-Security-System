# Face Detection Home Security System

This project implements a face detection home security system using Python and the `face_recognition` library. The system is designed to recognize members of your household and capture images of faces it does not recognize. The setup involves using a Raspberry Pi camera to monitor the surroundings and take appropriate actions.

## Features

- **Face Detection:** Detects faces in real-time using a webcam or Raspberry Pi camera.
- **Face Recognition:** Compares detected faces with known faces of household members.
- **Image Capture:** Captures images of unrecognized faces for further inspection.

## Requirements

- Python
- OpenCV
- NumPy
- face_recognition

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/face-detection-home-security.git
    cd face-detection-home-security
    ```

2. **Install the required Python packages:**

    ```bash
    pip install opencv-python numpy face_recognition
    ```

3. **Add images of known faces to the 'faces' directory.**

4. **Run the face recognition system:**

    ```bash
    python face_detection.py
    ```

## Configuration

- Adjust the `interval` parameter in the `FaceRecognition` class to control the frequency of face recognition.
- Fine-tune the `face_match_threshold` to set the confidence level for face recognition.

## Contributing

Feel free to contribute to this project by adding new features, fixing bugs, or improving documentation. Follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make changes and commit them.
4. Push the changes to your fork.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
