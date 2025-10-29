import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# IS_CLOUD = True
IS_CLOUD = "STREAMLIT_RUNTIME" in os.environ

# # ✅ Add the zoom CSS
# st.markdown("""
# <style>
# body {
#     zoom: 0.9;  /* Slight zoom effect */
#     transform-origin: 0 0;
# }
# </style>
# """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="Fire Detection System", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(45deg, #111F68, #042AFF);
        color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem;
        font-weight: bold; transition: all 0.3s ease; width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #042AFF, #111F68);
        transform: translateY(-2px); box-shadow: 0 4px 8px rgba(4, 42, 255, 0.3);
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
    Experience Real-Time Fire Detection on your Webcam, Videos, and Images | AI Monitoring and Alert System 🚀
    </h5>
</div>
"""

# --- Display titles in Streamlit ---
st.markdown(main_title_cfg, unsafe_allow_html=True)
st.markdown(sub_title_cfg, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    try:
        # st.image("techtics.png", width =275)
        st.image("logo.png", width=275)
    except:
        st.markdown("### 🔥 Fire Detection System")

    st.sidebar.title("User Configuration")
    source = st.selectbox("**Select Input Source**", ("Video", "Image", "Webcam"), index=0)
    confidence_threshold = st.slider("**Confidence Threshold**", 0.0, 1.0, 0.5, 0.05)

    uploaded_files = None
    if source == "Video":
        uploaded_files = st.file_uploader("**Choose a video file**", type=['mp4', 'avi', 'mov', 'mkv'],
                                          accept_multiple_files=False)
    elif source == "Image":
        uploaded_files = st.file_uploader("**Choose image files**", type=['jpg', 'jpeg', 'png', 'bmp'],
                                          accept_multiple_files=True)

    alert_threshold = st.slider("**Alert Threshold**", 0.0, 1.0, 0.7, 0.05)
    enable_alerts, show_stats = st.columns(2)
    with enable_alerts:
        enable_alerts = st.checkbox("**Enable Alerts**", value=True)
    with show_stats:
        show_stats = st.checkbox("**Show Statistics**", value=True)

    # Actions
    st.markdown("**Actions**")
    if source == "Video" and uploaded_files is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Start", key="start_video", use_container_width=True):
                st.session_state.start_video_processing = True
                st.rerun()
        with col2:
            if st.session_state.get("start_video_processing", False):
                if st.button("Stop", key="stop_video", use_container_width=True):
                    st.session_state.start_video_processing = False
                    st.rerun()
    elif source == "Image" and uploaded_files is not None and len(uploaded_files) > 0:
        if st.button("Detect Fire in Images", type="primary", key="start_image", use_container_width=True):
            st.session_state.start_image_processing = True
    elif source == "Webcam":

        if IS_CLOUD:
            st.warning("⚠️ Webcam detection only works locally, not on Streamlit Cloud.")
            # st.stop()
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Start", key="start_webcam", use_container_width=True):
                    st.session_state.run_webcam = True
                    st.rerun()
            with col2:
                if st.session_state.get("run_webcam", False):
                    if st.button("Stop", key="stop_webcam", use_container_width=True):
                        st.session_state.run_webcam = False
                        st.rerun()


# Load model
@st.cache_resource
def load_model():
    try:
        model = YOLO('fire_model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


model = load_model()
if model is None:
    st.error(
        "❌ Failed to load the fire detection model. Please check if 'fire_model.pt' exists in the current directory.")
    st.stop()


def preprocess_image(image_np):
    """Preprocess image to ensure it's in the correct format for the model"""
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB image
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_np


def process_frame(frame, confidence_threshold):
    try:
        # Ensure frame is in correct format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        original_frame = frame.copy()
        processing_frame = cv2.resize(frame, (640, 480))
        results = model(processing_frame, stream=True)

        annotated_frame = original_frame.copy()
        frame_has_fire, frame_max_confidence, fire_count = False, 0.0, 0

        for info in results:
            if info.boxes is not None:
                for box in info.boxes:
                    confidence = float(box.conf[0])
                    if confidence >= confidence_threshold:
                        frame_has_fire, fire_count = True, fire_count + 1
                        frame_max_confidence = max(frame_max_confidence, confidence)

                        x1, y1, x2, y2 = box.xyxy[0]
                        scale_x, scale_y = original_frame.shape[1] / 640, original_frame.shape[0] / 480
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                        # Blue bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                        # Text inside bounding box
                        label = f'FIRE {round(confidence * 100, 1)}%'
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                        # Position text inside the top of bounding box
                        text_x = x1 + 5
                        text_y = y1 + text_height + 5

                        # Draw background for text
                        cv2.rectangle(annotated_frame,
                                      (x1, y1),
                                      (x1 + text_width + 10, y1 + text_height + 10),
                                      (255, 0, 0), -1)

                        # Add white text inside bounding box
                        cv2.putText(annotated_frame, label, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame, frame_has_fire, frame_max_confidence, fire_count

    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        error_frame = frame.copy()
        cv2.putText(error_frame, f"Processing Error: {str(e)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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

st.session_state.source_selection = source

# Main processing
if source == "Video" and uploaded_files is not None:
    st.markdown("###  Video Fire Detection")

    if ('video_file' not in st.session_state or
            st.session_state.get('current_video') != uploaded_files.name or
            st.session_state.get('video_processed', False)):
        st.session_state.video_file = uploaded_files
        st.session_state.current_video = uploaded_files.name
        st.session_state.video_processed = False

    if st.session_state.get("start_video_processing", False) and not st.session_state.get('video_processed', False):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(st.session_state.video_file.getvalue())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file.")
            st.session_state.start_video_processing = False
            if os.path.exists(video_path):
                os.unlink(video_path)
            st.stop()

        col1, col2 = st.columns(2)
        original_placeholder, annotated_placeholder = col1.empty(), col2.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if show_stats:
            st.info(f"📊 Video Info: {total_frames} frames, {fps:.1f} FPS")

        progress_bar, status_text, alert_container = st.progress(0), st.empty(), st.empty()
        frame_count, fire_detections, max_confidence, total_fire_instances = 0, 0, 0.0, 0

        try:
            while (st.session_state.get("start_video_processing", False) and
                   st.session_state.get("source_selection") == "Video" and
                   not st.session_state.get('video_processed', False)):

                ret, frame = cap.read()
                if not ret:
                    st.session_state.video_processed = True
                    break

                frame_count += 1
                annotated_frame, frame_has_fire, frame_max_confidence, fire_count = process_frame(frame,
                                                                                                  confidence_threshold)

                if frame_has_fire:
                    fire_detections += 1
                    total_fire_instances += fire_count
                    max_confidence = max(max_confidence, frame_max_confidence)

                original_placeholder.image(frame, channels="BGR",
                                           caption=f"Original Frame - {frame.shape[1]}x{frame.shape[0]}")
                annotated_placeholder.image(annotated_frame, channels="BGR",
                                            caption=f"Fire Detection - {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")
                progress_bar.progress(frame_count / total_frames)

                if show_stats:
                    current_time = frame_count / fps if fps > 0 else 0
                    total_time = total_frames / fps if fps > 0 else 0
                    status_text.text(
                        f"⏱️ Time: {current_time:.1f}s / {total_time:.1f}s | Fire detected: {fire_detections} frames")

                if enable_alerts and frame_max_confidence >= alert_threshold:
                    alert_container.warning(f"🚨 HIGH FIRE ALERT! Confidence: {frame_max_confidence:.2f}")
                elif enable_alerts:
                    alert_container.empty()

        except Exception as e:
            st.error(f"Error during video processing: {str(e)}")
        finally:
            cap.release()
            if os.path.exists(video_path):
                os.unlink(video_path)
            st.session_state.start_video_processing = False
            st.session_state.video_processed = True
            st.success("✅ Video processing completed!")

            if show_stats:
                cols = st.columns(3)
                metrics = [("Total Frames", frame_count), ("Fire Detections", fire_detections),
                           ("Total Fire Instances", total_fire_instances)]
                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)

                cols = st.columns(3)
                detection_rate = (fire_detections / frame_count) * 100 if frame_count > 0 else 0
                instances_per_frame = total_fire_instances / frame_count if frame_count > 0 else 0
                metrics = [("Max Confidence", f"{max_confidence:.2f}"),
                           ("Fire Detection Rate", f"{detection_rate:.1f}%"),
                           ("Avg. Instances/Frame", f"{instances_per_frame:.1f}")]
                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)

elif source == "Image" and uploaded_files is not None and len(uploaded_files) > 0:
    st.markdown("### Image Fire Detection")
    if st.session_state.get("start_image_processing", False):
        images_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        for i, uploaded_file in enumerate(images_to_process):
            st.markdown(f"---")
            st.markdown(f"### Image {i + 1}: {uploaded_file.name}")
            try:
                # Convert uploaded file to image and preprocess
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_np = preprocess_image(image_np)

                # Process image for fire detection
                annotated_image, has_fire, max_confidence, fire_count = process_frame(image_np, confidence_threshold)

                # Convert back to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                original_image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) if len(image_np.shape) == 3 else image_np

                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image_rgb,
                             caption=f"Original Image - {original_image_rgb.shape[1]}x{original_image_rgb.shape[0]}",
                             use_container_width=True)
                with col2:
                    st.image(annotated_image_rgb,
                             caption=f"Fire Detection - {annotated_image_rgb.shape[1]}x{annotated_image_rgb.shape[0]}",
                             use_container_width=True)

                if show_stats:
                    st.markdown("#### 📊 Detection Results")
                    cols = st.columns(3)
                    status = "🔥 FIRE DETECTED" if has_fire else "✅ NO FIRE"
                    confidence_percent = round(max_confidence * 100, 1) if has_fire else 0
                    metrics = [("Status", status), ("Fire Instances", fire_count),
                               ("Max Confidence", f"{confidence_percent}%")]
                    for col, (label, value) in zip(cols, metrics):
                        col.metric(label, value)

                    if enable_alerts and has_fire and max_confidence >= alert_threshold:
                        st.error(f"🚨 FIRE ALERT! Confidence: {confidence_percent}%")
                    elif has_fire:
                        st.warning(f"⚠️ Fire detected with confidence: {confidence_percent}%")
                    else:
                        st.success("✅ No fire detected")

            except Exception as e:
                st.error(f"❌ Error processing image {uploaded_file.name}: {str(e)}")
                try:
                    st.image(Image.open(uploaded_file), caption=f"Original Image - Error: {str(e)}",
                             use_container_width=True)
                except:
                    st.write(f"Could not display image: {uploaded_file.name}")

        st.session_state.start_image_processing = False

elif source == "Webcam":
    st.markdown("### Live Webcam Fire Detection")
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access webcam.")
            st.session_state.run_webcam = False
            st.stop()

        col1, col2 = st.columns(2)
        original_placeholder, annotated_placeholder = col1.empty(), col2.empty()
        alert_container, stats_container = st.empty(), st.empty()
        frame_count, fire_detections, max_confidence = 0, 0, 0.0

        try:
            while st.session_state.run_webcam and st.session_state.get("source_selection") == "Webcam":
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read from webcam")
                    break

                frame_count += 1
                annotated_frame, frame_has_fire, frame_max_confidence, fire_count = process_frame(frame,
                                                                                                  confidence_threshold)

                if frame_has_fire:
                    fire_detections += 1
                    max_confidence = max(max_confidence, frame_max_confidence)

                original_placeholder.image(frame, channels="BGR",
                                           caption=f"Original Frame - {frame.shape[1]}x{frame.shape[0]}")
                annotated_placeholder.image(annotated_frame, channels="BGR",
                                            caption=f"Fire Detection - {annotated_frame.shape[1]}x{annotated_frame.shape[0]}")

                if show_stats:
                    detection_rate = (fire_detections / frame_count) * 100 if frame_count > 0 else 0
                    stats_container.text(
                        f"📊 Frames: {frame_count} | Fire Detections: {fire_detections} | Rate: {detection_rate:.1f}% | Max Confidence: {max_confidence:.2f}")

                if enable_alerts and frame_max_confidence >= alert_threshold:
                    alert_container.warning(f"🚨 HIGH FIRE ALERT! Confidence: {frame_max_confidence:.2f}")
                elif enable_alerts:
                    alert_container.empty()

        except Exception as e:
            st.error(f"Error during webcam processing: {str(e)}")
        finally:
            cap.release()
            st.session_state.run_webcam = False
            st.success("✅ Live detection stopped!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #111F68; margin-top: 2rem;'>
    <p>Made by <a href='https://techtics.ai' target='_blank' style='color: #042AFF; text-decoration: none; font-weight: bold;'>Techtics.ai</a></p>
</div>
""", unsafe_allow_html=True)
