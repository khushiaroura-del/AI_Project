import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time

st.set_page_config(
    page_title="AI Road Safety System",
    layout="wide"
)

st.title("🚗 AI Road Safety System")
st.write("Upload a road video for live AI detection.")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------
# UPLOAD VIDEO
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    st.success("Video Uploaded Successfully")

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)

    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    progress_bar = st.progress(0)

    status_text = st.empty()

    frame_placeholder = st.empty()

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model(
            frame,
            conf=0.25,
            verbose=False
        )

        h, w = frame.shape[:2]

        front_left = int(w * 0.35)
        front_right = int(w * 0.65)

        danger = False
        alert = False

        for box in results[0].boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in [
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "person"
            ]:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            area = (x2 - x1) * (y2 - y1)

            cx = (x1 + x2) // 2

            if (
                front_left < cx < front_right
                and area > 5000
            ):

                danger = True

                color = (0, 0, 255)
                text = "DANGER"

            elif area > 2000:

                alert = True

                color = (0, 165, 255)
                text = "ALERT"

            else:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Lane guide lines

        cv2.line(
            frame,
            (front_left, 0),
            (front_left, h),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (front_right, 0),
            (front_right, h),
            (255, 255, 255),
            2
        )

        # Road status

        if danger:

            cv2.putText(
                frame,
                "DANGER AHEAD",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        elif alert:

            cv2.putText(
                frame,
                "ALERT",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                3
            )

        else:

            cv2.putText(
                frame,
                "SAFE ROAD",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        frame_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True
        )

        if total_frames > 0:

            progress_bar.progress(
                min(
                    frame_number / total_frames,
                    1.0
                )
            )

        status_text.text(
            f"Frame {frame_number}/{total_frames}"
        )

        time.sleep(1 / fps)

    cap.release()

    st.success("✅ Detection Complete")
