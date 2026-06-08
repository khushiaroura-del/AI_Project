import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Road Safety System", layout="wide")
st.title("🚗 AI Road Safety System (YOLOv8)")

# -------------------------
# LOAD MODEL (OPTIMIZED)
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------
# SOUND FUNCTION
# -------------------------
def play_beep():
    beep_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
    </audio>
    """
    st.markdown(beep_html, unsafe_allow_html=True)

# -------------------------
# UPLOAD VIDEO
# -------------------------
uploaded_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov"])

if "last_state" not in st.session_state:
    st.session_state.last_state = "safe"

if not uploaded_file:
    st.info("Please upload a video to start detection")

if uploaded_file:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for performance
        if frame_count % 2 != 0:
            continue

        h, w = frame.shape[:2]

        front_left = int(w * 0.35)
        front_right = int(w * 0.65)

        results = model(frame, conf=0.25, verbose=False)

        danger = False
        alert = False

        # -------------------------
        # DETECTIONS
        # -------------------------
        for box in results[0].boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in ["car", "truck", "bus", "motorcycle", "bicycle", "person"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) // 2

            # DANGER ZONE
            if front_left < cx < front_right and area > 5000:
                danger = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "DANGER", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ALERT ZONE
            elif area > 2000:
                alert = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "ALERT", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # -------------------------
        # ZONE LINES
        # -------------------------
        cv2.line(frame, (front_left, 0), (front_left, h), (255, 255, 255), 2)
        cv2.line(frame, (front_right, 0), (front_right, h), (255, 255, 255), 2)

        # -------------------------
        # STATE MANAGEMENT
        # -------------------------
        if danger:

            status_placeholder.error("🔴 DANGER AHEAD")

            cv2.putText(frame, "DANGER AHEAD", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if st.session_state.last_state != "danger":
                play_beep()
                st.session_state.last_state = "danger"

        elif alert:

            status_placeholder.warning("🟠 ALERT")

            cv2.putText(frame, "ALERT", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            if st.session_state.last_state != "alert":
                play_beep()
                st.session_state.last_state = "alert"

        else:

            status_placeholder.success("🟢 SAFE ROAD")

            cv2.putText(frame, "SAFE ROAD", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            st.session_state.last_state = "safe"

        # -------------------------
        # DISPLAY FRAME
        # -------------------------
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()

    st.success("Processing Complete ✅")