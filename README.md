# Fire Detection System

This project provides real-time fire detection using a YOLO model trained on a small custom dataset. It supports video files, webcam streams, and image uploads. The application is built with Streamlit and runs directly on the web for easy access and testing.

Live App:
[https://fire-detections.streamlit.app/](https://fire-detections.streamlit.app/)

## Features

### YOLO-Based Detection

* Uses a YOLO model fine-tuned on a small fire dataset.
* Detects fire regions in real time.
* Works on images, videos, and webcam input.

### Streamlit Web Interface

* Simple and responsive UI.
* Upload images or videos for instant inference.
* Option to use a live webcam feed.
* Displays bounding boxes and detection confidence.

### Warning and Alerts

* Shows warning messages when fire is detected.
* Highlights detected regions clearly.
* Designed for fast decision-making and safety monitoring.

### Demo Images

A set of images is available to quickly test the model's performance.

Link:
[https://fire-detections.streamlit.app/](https://fire-detections.streamlit.app/)

## Tech Stack

* Python
* Streamlit
* YOLO (fine-tuned on a small fire dataset)
* OpenCV (video and image processing)

## How It Works

1. User uploads an image, video, or enables webcam mode.
2. The frame is passed to the YOLO model for inference.
3. Fire objects are detected and highlighted with bounding boxes.
4. If fire is detected, a warning alert is displayed.
5. Output is shown in the browser for real-time results.

## Project Structure

```
app/
 ├── streamlit_app.py     # Main application
models/
 └── fire_yolo.pt         # Trained YOLO model
utils/
 ├── detection.py         # Inference functions
 └── processing.py        # Frame/image utilities
assets/
 └── demo_images/         # Test images
```

## Use Cases

* Home safety monitoring
* Industrial fire detection
* Surveillance systems
* Research and dataset testing
* Educational demonstrations

## Future Improvements

* Larger dataset for stronger generalization
* Smoke detection capabilities
* Multi-camera monitoring
* Notification system (SMS/Email)
* Deployment container (Docker)

## Contributing

Contributions and improvements are welcome.
Feel free to open issues or submit pull requests.
