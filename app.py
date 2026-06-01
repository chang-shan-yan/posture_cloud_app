# ==============================
# app.py (Streamlit Cloud Advanced AI Pose Version)
# ==============================

import streamlit as st
import numpy as np
import tempfile
import math
from gtts import gTTS

# -----------------------------
# Safe imports (Cloud stable)
# -----------------------------
try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
except Exception:
    mp = None
    pose = None


st.set_page_config(page_title="AI姿勢偵測進階版", layout="centered")

st.title("🧠 AI 姿勢偵測進階版（雲端穩定 + 骨架分析）")

st.write("支援圖片與影片的姿勢骨架分析（MediaPipe + 雲端優化）")

file = st.file_uploader("上傳圖片或影片", type=["jpg", "png", "mp4", "mov"])


# -----------------------------
# 語音提醒 (gTTS)
# -----------------------------

def speak(text):
    tts = gTTS(text=text, lang="zh-tw")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name


# -----------------------------
# 角度計算
# -----------------------------

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# -----------------------------
# 姿勢分析（核心AI）
# -----------------------------

def analyze_pose(image):
    if pose is None or image is None:
        return "MediaPipe不可用", image

    results = pose.process(image)

    if not results.pose_landmarks:
        return "未偵測到人體", image

    landmarks = results.pose_landmarks.landmark

    # 取得關鍵點（頭-肩-髖）
    try:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        neck_angle = calculate_angle(ear, shoulder, hip)

        # 判斷姿勢
        if neck_angle < 150:
            status = "⚠️ 頭部前傾（姿勢不良）"
        else:
            status = "✅ 姿勢良好"

        return status, results

    except Exception:
        return "分析失敗", results


# -----------------------------
# 影片處理（安全版）
# -----------------------------

def process_video(video_file):
    if cv2 is None:
        return "OpenCV不可用"

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_count = 0
    bad_posture = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 60:  # 限制60幀避免爆炸
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result, _ = analyze_pose(image)

        if "不良" in result:
            bad_posture += 1

        frame_count += 1

    cap.release()

    if bad_posture > 10:
        return "⚠️ 偵測到多次不良姿勢"
    else:
        return "✅ 姿勢整體良好"


# -----------------------------
# 主流程
# -----------------------------

if file is not None:

    file_type = file.type

    # 圖片
    if "image" in file_type:
        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)

        if cv2:
            image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = None

        st.image(image, caption="上傳圖片", use_container_width=True)

        status, _ = analyze_pose(image)

        st.success(status)

        audio = speak(status)
        st.audio(audio)

    # 影片
    elif "video" in file_type:
        st.video(file)

        st.info("AI正在分析影片姿勢（雲端限制：只分析前60幀）")

        result = process_video(file)
        st.write(result)

        audio = speak(result)
        st.audio(audio)

else:
    st.info("請上傳圖片或影片開始AI分析")
