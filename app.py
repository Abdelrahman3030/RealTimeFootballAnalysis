import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO  # YOLO import
from moviepy.editor import VideoFileClip
import gdown


download_url = "https://drive.google.com/file/d/1ehr7HiYSVPBQOx1JZx5aWTLynRC4Fa4E/view?usp=sharing"
output = "yolov5xu.pt"

# Check if the file already exists to avoid redownloading
if not os.path.exists(output):
    print("Downloading model yolov5xu.pt from Google Drive...")
    gdown.download(download_url, output, quiet=False)
else:
    print("Model already downloaded.")
    
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
            background-color: #2E2E2E;  /* Darker background for better contrast */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FF6347;  /* Tomato color for headings */
        }
        .stButton button {
            background-color: #FF6347;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px;
        }
        .stFileUploader {
            background-color: #FF6347;
            border: none;
            color: white;
            border-radius: 5px;
        }
        .stFileUploader > div {
            text-align: center;
        }
        .uploadedVideoText {
            font-size: 1.2rem;
            color: #FF6347;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #2E2E2E;
            text-align: center;
            padding: 10px;
        }
        .footer a {
            color: #FF6347;
            text-decoration: none;
            font-weight: bold;
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

        # If results are a list, process each result individually
        for result in results:
            result_frame = np.squeeze(result.plot())  # Use `.plot()` to draw bounding boxes

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
