
import streamlit as st
import cv2
import numpy as np
import time
import math
from gtts import gTTS
import tempfile
import mediapipe as mp

st.set_page_config(page_title="AI 姿勢偵測雲端版", layout="wide")
st.title("AI 肩頸烏龜頸與駝背偵測（雲端可部署版）")

# -----------------------
# utils
# -----------------------
def calculate_line_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(abs(dy), abs(dx)) * 180.0 / math.pi

def to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def speak(text):
    tts = gTTS(text=text, lang="zh-tw")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -----------------------
# sidebar
# -----------------------
with st.sidebar:
    st.header("參數")
    dist_thresh = st.slider("烏龜頸距離閾值", 5, 200, 60)
    angle_thresh = st.slider("駝背角度閾值", 30, 90, 55)
    st.markdown("---")
    st.info("雲端版：使用拍照偵測（避免攝影機限制）")

# -----------------------
# camera input (cloud safe)
# -----------------------
img_file = st.camera_input("拍攝你的姿勢（或使用鏡頭）")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    status = "正常"
    color = (0, 255, 0)
    alert_text = None

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        le = to_pixel(lm[7], w, h)
        re = to_pixel(lm[8], w, h)
        ls = to_pixel(lm[11], w, h)
        rs = to_pixel(lm[12], w, h)

        left_dx = le[0] - ls[0]
        right_dx = re[0] - rs[0]

        if abs(left_dx) > abs(right_dx):
            dist = left_dx
            angle = calculate_line_angle(le, ls)
        else:
            dist = right_dx
            angle = calculate_line_angle(re, rs)

        if dist > dist_thresh:
            status = "⚠️ 烏龜頸"
            color = (255, 165, 0)
            alert_text = "請收下巴！"

        if angle < angle_thresh:
            status = "🚨 駝背"
            color = (255, 0, 0)
            alert_text = "請挺胸坐直！"

        st.write(f"距離: {dist:.1f}")
        st.write(f"角度: {angle:.1f}")
        st.subheader(status)

        if alert_text:
            audio_path = speak(alert_text)
            st.audio(audio_path)

    st.image(frame, channels="BGR")

else:
    st.info("請拍攝一張照片開始偵測")
