
import streamlit as st
import numpy as np
import tempfile
from ultralytics import YOLO
import cv2
import os

# Page setup
st.set_page_config(page_title="Football Skill Evaluator", layout="centered")
st.title("⚽ أسطورة الغد | Legend of Tomorrow")
st.markdown("### Jumping with Ball Evaluation (Age ~8)")
st.write("Upload a video of the child performing jumping with the ball. The AI will detect knee touches and score the performance.")

# Upload video
video_file = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])

# Load YOLO models (cached to avoid reloading)
@st.cache_resource
def load_models():
    pose_model = YOLO("yolov8n-pose.pt")   # For pose/keypoints
    ball_model = YOLO("yolov8n.pt")        # For ball detection
    return pose_model, ball_model

pose_model, ball_model = load_models()

# Define evaluation logic
def detect_ball_knee_contacts(video_path, frame_skip=2, distance_thresh=40, angle_thresh=60):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    touch_count = 0
    successful_touches = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        ball_results = ball_model(frame)[0]
        ball_pos = None
        for box, cls in zip(ball_results.boxes.xyxy, ball_results.boxes.cls):
            if int(cls) == 32:  # Class 32 = ball in COCO
                x1, y1, x2, y2 = map(int, box)
                ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                break

        pose_results = pose_model(frame)[0]
        if pose_results.keypoints is not None and pose_results.keypoints.xy.shape[0] > 0:
            keypoints = pose_results.keypoints.xy.cpu().numpy()[0]
            if keypoints.shape[0] >= 17:
                hip = keypoints[11]
                knee = keypoints[13]
                ankle = keypoints[15]

                def angle(a, b, c):
                    a, b, c = np.array(a), np.array(b), np.array(c)
                    ang = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    return np.abs(np.degrees(ang)) % 180

                knee_angle = angle(hip, knee, ankle)

                if ball_pos:
                    dist = np.linalg.norm(np.array(ball_pos) - np.array(knee))
                    if dist < distance_thresh and knee_angle < angle_thresh:
                        touch_count += 1
                        successful_touches.append({
                            "frame": frame_idx,
                            "distance": round(dist, 1),
                            "angle": round(knee_angle, 1)
                        })

        frame_idx += 1

    cap.release()
    score = min(5, touch_count)
    return score, successful_touches

# Process and evaluate
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("🧠 Evaluating... please wait.")
    score, results = detect_ball_knee_contacts(video_path)

    st.success(f"🏆 Final Score: {score}/5")
    st.write("✅ Valid touches detected:")
    st.json(results)
