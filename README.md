# Fire Detection System

This project provides real-time fire detection using a YOLO model trained on a small custom dataset. The application runs entirely on the web using Streamlit and supports image uploads, video files, and webcam input. A warning alert is displayed whenever fire is detected. Demo media files are included for testing.

Live App:
[https://fire-detections.streamlit.app/](https://fire-detections.streamlit.app/)

## Features

### YOLO Fire Detection

* Uses a YOLO model (`fire_model.pt`) fine-tuned on a small fire dataset.
* Detects fire in real time across images, videos, and webcam streams.
* Output includes bounding boxes and confidence scores.

### Streamlit Web Interface

* Simple browser-based UI built with Streamlit.
* Supports:

  * Uploading photos
  * Uploading videos
  * Running webcam detection
* Automatically displays processed frames with detections.

### Warning Alerts

* Shows an alert message when fire is detected.
* Helps in quick decision-making and early monitoring.

### Demo Media Included

You can test the app using the provided sample files:

* `demo.jpg`
* `fire.mp4`

These allow quick evaluation of detection performance.

## Project Structure

```
.devcontainer/            # Development container config (optional)
streamlit/                # Additional Streamlit configurations (if any)
README.md                 # Documentation
demo.jpg                  # Sample test image
fire.mp4                  # Sample test video
fire_model.pt             # Trained YOLO fire detection model
logo.png                  # Application/logo asset
main.py                   # Main Streamlit application
packages.txt              # System-level packages (for Streamlit Cloud/Spaces)
requirements.txt          # Python dependencies
```

## Tech Stack

* Python
* Streamlit
* YOLO (fine-tuned model)
* OpenCV for frame and video processing

## How It Works

1. User uploads a photo/video or enables webcam mode.
2. The media is processed frame-by-frame using the YOLO model.
3. Detected fire regions are outlined with bounding boxes.
4. If any fire is found, a visible warning alert is shown.
5. Output is displayed on the web interface in real time.

## Installation (Local)

```
pip install -r requirements.txt
streamlit run main.py
```

## Use Cases

* Home/office surveillance
* Factory or warehouse monitoring
* Early fire detection in remote environments
* Research and educational demonstration

## Future Improvements

* Larger dataset training for higher accuracy
* Smoke detection support
* Notification system (email/SMS/API alerts)
* Faster model version (quantized/mobile)
* Docker deployment

## Contributing

Contributions are welcome.
Open an issue or submit a pull request for enhancements.
