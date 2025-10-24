import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with bluish theme
st.markdown("""
<style>
    .main-header {
        color: #111F68;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #042AFF;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .detection-box {
        border: 3px solid #111F68;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background: linear-gradient(135deg, #F0F4FF 0%, #E8EDFF 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-high {
        color: #FF0000;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-medium {
        color: #FF8C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #FFA500;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1A1F4D 0%, #111F68 100%);
        color: white;
    }
    .sidebar .sidebar-content .stSelectbox > div > div {
        background-color: #2A2F6D;
        color: white;
    }
    .sidebar .sidebar-content .stSlider > div > div {
        background-color: #2A2F6D;
    }
    .sidebar .sidebar-content .stCheckbox > div > div {
        background-color: #2A2F6D;
    }
    /* Bluish theme for buttons */
    .stButton > button {
        background: linear-gradient(45deg, #111F68, #042AFF);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #042AFF, #111F68);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(4, 42, 255, 0.3);
    }
    /* Specific styling for start button */
    .start-button > button {
        background: linear-gradient(45deg, #111F68, #042AFF);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .start-button > button:hover {
        background: linear-gradient(45deg, #042AFF, #111F68);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(4, 42, 255, 0.3);
    }
    /* Specific styling for stop button */
    .stop-button > button {
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stop-button > button:hover {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 75, 43, 0.3);
    }
    .compact-button {
        width: 100% !important;
        margin: 5px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Main title ---
main_title_cfg = """
<div>
    <h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
    font-family: 'Archivo', sans-serif; margin-bottom:20px;">
    🔥 Fire Detection System 🔥
    </h1>
</div>
"""

# --- Subtitle ---
sub_title_cfg = """
<div>
    <h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
    margin-top:-15px; margin-bottom:50px;">
    Experience Real-Time Fire Detection on your Webcam, Videos, and Images | AI Monitoring and Alert System
    </h5>
</div>
"""

# --- Display titles in Streamlit ---
st.markdown(main_title_cfg, unsafe_allow_html=True)
st.markdown(sub_title_cfg, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Logo
    try:
        st.image("Techtics-Logo-Light.png", width=250)
    except:
        st.markdown("### 🔥 Fire Detection System")

    # Main heading
    st.sidebar.title("User Configuration")

    # Input source
    source = st.selectbox(
        "**Select Input Source**",
        ("Video", "Image", "Webcam"),
        help="Choose between uploading a video file, image, or using your webcam"
    )

    # Confidence threshold
    confidence_threshold = st.slider(
        "**Confidence Threshold**",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for fire detection"
    )

    # Always-on display options (hidden but active)
    show_original = True
    show_annotated = True

    # File uploaders
    uploaded_files = None
    if source == "Video":
        uploaded_files = st.file_uploader(
            "**Choose a video file**",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for fire detection",
            accept_multiple_files=False
        )
    elif source == "Image":
        uploaded_files = st.file_uploader(
            "**Choose image files**",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload one or more image files to analyze for fire detection",
            accept_multiple_files=True
        )

    # Alert threshold slider (below them)
    alert_threshold = st.slider(
        "**Alert Threshold**",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Confidence threshold for triggering alerts"
    )

    # Alert settings
    col1, col2 = st.columns(2)
    with col1:
        enable_alerts = st.checkbox("**Enable Alerts**", value=True)
    with col2:
        show_stats = st.checkbox("**Show Statistics**", value=True)

    # === Actions ===
    st.markdown("**Actions**")

    # VIDEO MODE
    if source == "Video" and uploaded_files is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Start", key="start_video", use_container_width=True):
                st.session_state.start_video_processing = True
                # Reset video state when starting
                if 'video_path' in st.session_state:
                    if os.path.exists(st.session_state.video_path):
                        os.unlink(st.session_state.video_path)
                    del st.session_state.video_path
                if 'cap' in st.session_state:
                    if st.session_state.cap.isOpened():
                        st.session_state.cap.release()
                    del st.session_state.cap
                st.rerun()

        with col2:
            if st.session_state.get("start_video_processing", False):
                if st.button("Stop", key="stop_video", use_container_width=True):
                    st.session_state.start_video_processing = False
                    # Clean up resources
                    if 'cap' in st.session_state:
                        if st.session_state.cap.isOpened():
                            st.session_state.cap.release()
                        del st.session_state.cap
                    st.rerun()

    # IMAGE MODE
    elif source == "Image" and uploaded_files is not None and len(uploaded_files) > 0:
        if st.button("Detect Fire in Images", type="primary", key="start_image", use_container_width=True):
            st.session_state.start_image_processing = True

    # WEBCAM MODE
    elif source == "Webcam":
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Start", key="start_webcam", use_container_width=True):
                st.session_state.run_webcam = True
                st.rerun()

        with col2:
            if st.session_state.get("run_webcam", False):
                if st.button("Stop", key="stop_webcam", use_container_width=True):
                    st.session_state.run_webcam = False
                    if 'webcam_cap' in st.session_state:
                        if st.session_state.webcam_cap.isOpened():
                            st.session_state.webcam_cap.release()
                        del st.session_state.webcam_cap
                    st.rerun()


# Load the fire detection model
@st.cache_resource
def load_model():
    """Load the fire detection model"""
    try:
        model = YOLO('fire_model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Initialize model
model = load_model()

if model is None:
    st.error(
        "❌ Failed to load the fire detection model. Please check if 'fire_model.pt' exists in the current directory.")
    st.stop()


def preprocess_image(image_np):
    """Preprocess image to ensure it's in the correct format for the model"""
    # Convert grayscale to RGB if needed
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_np


def get_display_size(frame):
    """Calculate appropriate display size for frames"""
    height, width = frame.shape[:2]

    # If image is too large, scale it down
    if width > 1200 or height > 800:
        scale = min(1200 / width, 800 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return new_width, new_height
    else:
        return width, height


def process_frame(frame, confidence_threshold):
    """Process a single frame for fire detection"""
    try:
        # Ensure frame is in correct format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Keep original frame for display
        original_frame = frame.copy()

        # Resize frame for processing (model expects consistent size)
        processing_frame = cv2.resize(frame, (640, 480))

        # Run inference
        results = model(processing_frame, stream=True)

        # Process detections on original frame
        annotated_frame = original_frame.copy()
        frame_has_fire = False
        frame_max_confidence = 0.0
        fire_count = 0

        for info in results:
            boxes = info.boxes
            if boxes is not None:
                for box in boxes:
                    # FIX: Convert tensor to float first
                    confidence = float(box.conf[0])  # Convert tensor to float
                    # FIX: Use exact percentage without rounding up
                    confidence_percent = round(confidence * 100, 1)  # Show 1 decimal place

                    # Use >= instead of > for confidence threshold
                    if confidence >= confidence_threshold:
                        frame_has_fire = True
                        fire_count += 1
                        frame_max_confidence = max(frame_max_confidence, confidence)

                        # Get bounding box coordinates (scale from processed to original)
                        x1, y1, x2, y2 = box.xyxy[0]

                        # Scale coordinates back to original size
                        scale_x = original_frame.shape[1] / 640
                        scale_y = original_frame.shape[0] / 480

                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

                        # Add confidence text
                        label = f'FIRE {confidence_percent}%'
                        cv2.putText(annotated_frame, label, (x1 + 8, y1 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated_frame, frame_has_fire, frame_max_confidence, fire_count

    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        # Return original frame with error info
        error_frame = frame.copy()
        cv2.putText(error_frame, f"Processing Error: {str(e)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_frame, False, 0.0, 0


# Initialize session states
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "start_video_processing" not in st.session_state:
    st.session_state.start_video_processing = False
if "start_image_processing" not in st.session_state:
    st.session_state.start_image_processing = False
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# Store current source in session state
st.session_state.source_selection = source

# Main content area based on source selection
if source == "Video":
    if uploaded_files is not None:
        st.markdown("### 📹 Video Fire Detection")

        # Store video file in session state to prevent re-upload issues
        if ('video_file' not in st.session_state or
                st.session_state.get('current_video') != uploaded_files.name or
                st.session_state.get('video_processed', False)):

            st.session_state.video_file = uploaded_files
            st.session_state.current_video = uploaded_files.name
            st.session_state.video_processed = False
            # Clean up any existing video processing state
            if 'cap' in st.session_state:
                if st.session_state.cap.isOpened():
                    st.session_state.cap.release()
                del st.session_state.cap
            if 'video_path' in st.session_state:
                if os.path.exists(st.session_state.video_path):
                    os.unlink(st.session_state.video_path)
                del st.session_state.video_path

        if st.session_state.get("start_video_processing", False) and not st.session_state.get('video_processed', False):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(st.session_state.video_file.getvalue())
                video_path = tmp_file.name
                st.session_state.video_path = video_path

            # Create columns for display
            if show_original and show_annotated:
                col1, col2 = st.columns(2)
            else:
                col1 = st.container()
                col2 = None

            # Initialize video capture with error handling
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("❌ Could not open video file. The file may be corrupted or in an unsupported format.")
                    st.session_state.start_video_processing = False
                    if os.path.exists(video_path):
                        os.unlink(video_path)
                    st.stop()

                st.session_state.cap = cap  # Store in session state for forced stop

            except Exception as e:
                st.error(f"❌ Error opening video file: {str(e)}")
                st.session_state.start_video_processing = False
                if os.path.exists(video_path):
                    os.unlink(video_path)
                st.stop()

            # Create placeholders for frames
            if show_original:
                original_placeholder = col1.empty()
            if show_annotated:
                annotated_placeholder = col2.empty() if col2 else st.empty()

            # Statistics
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            if show_stats:
                st.info(f"📊 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Alert container
            alert_container = st.empty()

            frame_count = 0
            fire_detections = 0
            max_confidence = 0.0
            total_fire_instances = 0

            # Process video frames
            try:
                while (st.session_state.get("start_video_processing", False) and
                       st.session_state.get("source_selection") == "Video" and
                       not st.session_state.get('video_processed', False)):

                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.video_processed = True
                        break

                    frame_count += 1

                    # Process frame
                    annotated_frame, frame_has_fire, frame_max_confidence, fire_count = process_frame(frame,
                                                                                                      confidence_threshold)

                    if frame_has_fire:
                        fire_detections += 1
                        total_fire_instances += fire_count
                        max_confidence = max(max_confidence, frame_max_confidence)

                    # Display frames with size info
                    if show_original:
                        original_placeholder.image(frame, channels="BGR",
                                                   caption=f"Original Frame - {frame.shape[1]}x{frame.shape[0]}")
                    if show_annotated:
                        annotated_placeholder.image(annotated_frame, channels="BGR",
                                                    caption=f"Fire Detection - {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")

                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)

                    # Calculate time information
                    current_time = frame_count / fps if fps > 0 else 0
                    total_time = total_frames / fps if fps > 0 else 0

                    if show_stats:
                        status_text.text(
                            f"⏱️ Time: {current_time:.1f}s / {total_time:.1f}s | Fire detected: {fire_detections} frames")

                    # Alert for high confidence detections
                    if enable_alerts and frame_max_confidence >= alert_threshold:
                        alert_container.warning(f"🚨 HIGH FIRE ALERT! Confidence: {frame_max_confidence:.2f}")
                    elif enable_alerts and alert_container:
                        alert_container.empty()

            except Exception as e:
                st.error(f"Error during video processing: {str(e)}")

            finally:
                # Clean up
                if 'cap' in st.session_state:
                    if st.session_state.cap.isOpened():
                        st.session_state.cap.release()
                    del st.session_state.cap
                if 'video_path' in st.session_state:
                    if os.path.exists(st.session_state.video_path):
                        os.unlink(st.session_state.video_path)
                    del st.session_state.video_path

                # Reset processing state
                st.session_state.start_video_processing = False
                st.session_state.video_processed = True

                # Final statistics
                st.success("✅ Video processing completed!")

                if show_stats:
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Frames", frame_count)
                    with col2:
                        st.metric("Fire Detections", fire_detections)
                    with col3:
                        st.metric("Total Fire Instances", total_fire_instances)

                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Max Confidence", f"{max_confidence:.2f}")
                    with col5:
                        detection_rate = (fire_detections / frame_count) * 100 if frame_count > 0 else 0
                        st.metric("Fire Detection Rate", f"{detection_rate:.1f}%")
                    with col6:
                        instances_per_frame = total_fire_instances / frame_count if frame_count > 0 else 0
                        st.metric("Avg. Instances/Frame", f"{instances_per_frame:.1f}")

    else:
        st.info("📹 Please upload a video file to start fire detection")

elif source == "Image":
    if uploaded_files is not None and len(uploaded_files) > 0:
        st.markdown("### 📷 Image Fire Detection")

        # Process single or multiple images
        images_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

        if st.session_state.get("start_image_processing", False):
            for i, uploaded_file in enumerate(images_to_process):
                st.markdown(f"---")
                st.markdown(f"### Image {i + 1}: {uploaded_file.name}")

                try:
                    # Convert uploaded file to image
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)

                    # Preprocess image for model compatibility
                    image_np = preprocess_image(image_np)

                    # Process image
                    annotated_image, has_fire, max_confidence, fire_count = process_frame(image_np,
                                                                                          confidence_threshold)

                    # Convert back to RGB for display
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    original_image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if len(
                        image_np.shape) == 3 else image_np

                    # Display results with original size info
                    if show_original and show_annotated:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_image_rgb,
                                     caption=f"Original Image - {original_image_rgb.shape[1]}x{original_image_rgb.shape[0]}",
                                     use_container_width=True)
                        with col2:
                            st.image(annotated_image_rgb,
                                     caption=f"Fire Detection - {annotated_image_rgb.shape[1]}x{annotated_image_rgb.shape[0]}",
                                     use_container_width=True)
                    elif show_original:
                        st.image(original_image_rgb,
                                 caption=f"Original Image - {original_image_rgb.shape[1]}x{original_image_rgb.shape[0]}",
                                 use_container_width=True)
                    elif show_annotated:
                        st.image(annotated_image_rgb,
                                 caption=f"Fire Detection - {annotated_image_rgb.shape[1]}x{annotated_image_rgb.shape[0]}",
                                 use_container_width=True)

                    # Display statistics
                    if show_stats:
                        st.markdown("#### 📊 Detection Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            status = "🔥 FIRE DETECTED" if has_fire else "✅ NO FIRE"
                            st.metric("Status", status)
                        with col2:
                            st.metric("Fire Instances", fire_count)
                        with col3:
                            # FIX: Show exact percentage without rounding up
                            confidence_percent = round(max_confidence * 100, 1) if has_fire else 0
                            st.metric("Max Confidence", f"{confidence_percent}% ({max_confidence:.2f})")

                        # Alert - Use >= for alert threshold
                        if enable_alerts and has_fire and max_confidence >= alert_threshold:
                            st.error(f"🚨 FIRE ALERT! Confidence: {confidence_percent}%")
                        elif has_fire:
                            st.warning(f"⚠️ Fire detected with confidence: {confidence_percent}%")
                        else:
                            st.success("✅ No fire detected")

                except Exception as e:
                    st.error(f"❌ Error processing image {uploaded_file.name}: {str(e)}")
                    # Display the original image with error message
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Original Image - Error: {str(e)}", use_container_width=True)
                    except:
                        st.write(f"Could not display image: {uploaded_file.name}")

                    continue  # Continue with next image

            # Reset processing state after completion
            st.session_state.start_image_processing = False
    else:
        st.info("📷 Please upload one or more image files to analyze")

elif source == "Webcam":
    st.markdown("### 📷 Live Webcam Fire Detection")

    if st.session_state.run_webcam:
        # Create columns for display
        if show_original and show_annotated:
            col1, col2 = st.columns(2)
        else:
            col1 = st.container()
            col2 = None

        cap = cv2.VideoCapture(0)
        st.session_state.webcam_cap = cap  # Store in session state for forced stop

        if not cap.isOpened():
            st.error("❌ Could not access webcam.")
            st.session_state.run_webcam = False
            if 'webcam_cap' in st.session_state:
                del st.session_state.webcam_cap
            st.stop()

        if show_original:
            original_placeholder = col1.empty()
        if show_annotated:
            annotated_placeholder = col2.empty() if col2 else st.empty()

        # Alert container
        alert_container = st.empty()
        stats_container = st.empty()

        frame_count = 0
        fire_detections = 0
        max_confidence = 0.0

        # Webcam processing loop
        try:
            while (st.session_state.run_webcam and
                   st.session_state.get("source_selection") == "Webcam"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read from webcam")
                    break

                frame_count += 1

                # Process frame
                annotated_frame, frame_has_fire, frame_max_confidence, fire_count = process_frame(frame,
                                                                                                  confidence_threshold)

                if frame_has_fire:
                    fire_detections += 1
                    max_confidence = max(max_confidence, frame_max_confidence)

                # Display frames with size info
                if show_original:
                    original_placeholder.image(frame, channels="BGR",
                                               caption=f"Original Frame - {frame.shape[1]}x{frame.shape[0]}")
                if show_annotated:
                    annotated_placeholder.image(annotated_frame, channels="BGR",
                                                caption=f"Fire Detection - {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")

                # Update statistics
                if show_stats:
                    detection_rate = (fire_detections / frame_count) * 100 if frame_count > 0 else 0
                    stats_container.text(
                        f"📊 Frames: {frame_count} | Fire Detections: {fire_detections} | Rate: {detection_rate:.1f}% | Max Confidence: {max_confidence:.2f}")

                # Alert for high confidence detections
                if enable_alerts and frame_max_confidence >= alert_threshold:
                    alert_container.warning(f"🚨 HIGH FIRE ALERT! Confidence: {frame_max_confidence:.2f}")
                elif enable_alerts and alert_container:
                    alert_container.empty()

        except Exception as e:
            st.error(f"Error during webcam processing: {str(e)}")

        finally:
            # Immediately release camera when stopped
            cap.release()
            if 'webcam_cap' in st.session_state:
                del st.session_state.webcam_cap
            st.session_state.run_webcam = False
            st.success("✅ Live detection stopped!")

# Footer with bluish theme
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #111F68; margin-top: 2rem;'>
        <p>Made by 
            <a href='https://techtics.ai' target='_blank' 
            style='color: #042AFF; text-decoration: none; font-weight: bold;'>
            Techtics.ai
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)