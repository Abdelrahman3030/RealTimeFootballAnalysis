import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO  # YOLO import
from moviepy.editor import VideoFileClip

# Load YOLOv5 model (using your custom-trained model)
model = YOLO('yolov5xu.pt')  # Load your custom YOLOv5 model (yolov5xu.pt or yolov5/best.pt)

# Set page configuration
st.set_page_config(
    page_title="Real-Time Sports Analytics",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom styling for the Streamlit app
st.markdown("""
    <style>
        .main {
            background-color: #F0F2F6;
        }
        h1 {
            color: #FF6347;
        }
        .stButton button {
            background-color: #FF6347;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("⚽ Real-Time Sports Analytics with YOLOv5")
st.subheader("Object Detection and Player Tracking on Videos")

# Video uploader
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Process the video using MoviePy
    video = VideoFileClip(tfile.name)

    # Create a placeholder for displaying video frames in Streamlit
    stframe = st.empty()

    # Process each frame of the video
    for frame in video.iter_frames(fps=24, dtype="uint8"):  # Extract frames from video
        # YOLOv5 inference on each frame
        results = model(frame)  # Process the frame through YOLOv5

        # If results is a list, render each result individually
        if isinstance(results, list):
            for result in results:
                result_frame = np.squeeze(result.render())  # Draw bounding boxes on the frame
        else:
            # If not a list, handle it directly
            result_frame = np.squeeze(results.render())  # Draw bounding boxes on the frame

        # Display the processed frame in Streamlit (MoviePy uses RGB by default)
        stframe.image(result_frame, channels="RGB", use_column_width=True)

    st.write("### Video processing complete.")
else:
    st.write("Upload a video to start processing.")

# Footer for the app
st.markdown("<hr>", unsafe_allow_html=True)
st.write("""
    *Developed using [Streamlit](https://www.streamlit.io/), YOLOv5, and MoviePy.*
""")