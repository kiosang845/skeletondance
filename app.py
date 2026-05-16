"""
舞迹 - AI舞蹈扒舞与成长记录系统
功能：参考视频骨架提取、摄像头录制或本地视频上传、时间对齐、对比打分、偏差标注、历史记录。
新增：脊柱/胸腔骨架、视频宽度控制、细节放大（手/脚）、时间段循环播放。
技术栈：Streamlit + MediaPipe + OpenCV + Streamlit-WebRTC
所有处理均在本地完成，不传输任何数据到外部。
"""

import sys
import os
import re
import glob
import time
import json
import threading
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque

import av
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec, _normalized_to_pixel_coordinates

import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
    RTCConfiguration,
)
from scipy.spatial.distance import euclidean
from scipy.signal import argrelextrema
from plotly import graph_objs as go
from PIL import Image

# 用于自定义 HTML 组件
import streamlit.components.v1 as components

# ------------------------- 配置与初始化 -------------------------
st.set_page_config(page_title="舞迹 - AI舞蹈扒舞与成长系统", layout="wide")

# 创建数据存储目录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SKELETON_DIR = DATA_DIR / "skeletons"
SKELETON_DIR.mkdir(exist_ok=True)
VIDEO_DIR = DATA_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.json"
if not HISTORY_FILE.exists():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# MediaPipe Pose 初始化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# 关节索引与名称（33个关键点）
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# 评分使用的关节分组
JOINT_GROUPS = {
    "左臂": [11, 13, 15],
    "右臂": [12, 14, 16],
    "左腿": [23, 25, 27],
    "右腿": [24, 26, 28],
    "躯干": [11, 12, 23, 24],
    "头部": [0, 7, 8],
}

# 颜色定义
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)

# ------------------------- 工具函数 -------------------------
@st.cache_data(show_spinner="正在提取骨架...")
def extract_skeleton_from_video(video_path: str, fps: int = None, mirror: bool = False) -> Tuple[List, int]:
    """离线提取视频中每一帧的骨架关键点(33个,归一化坐标)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = orig_fps
    skeletons = []
    frame_idx = 0
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            if fps < orig_fps and frame_idx % round(orig_fps / fps) != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                skeletons.append(landmarks)
            else:
                skeletons.append(None)
    cap.release()
    return skeletons, fps

# ---------- 美观骨架绘制（含脊柱与胸腔） ----------
def draw_skeleton_pretty(image, landmarks):
    """绘制美观骨架：躯干四肢连线、脊柱线、胸腔轮廓、头部椭圆。"""
    if not landmarks:
        return
    h, w, _ = image.shape

    def pt(idx):
        if idx >= len(landmarks) or landmarks[idx] is None:
            return None
        return int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)

    TRUNK_LIMB_CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    SKELETON_COLORS = {
        "torso": (255, 128, 0),
        "left_arm": (0, 255, 0),
        "right_arm": (0, 255, 255),
        "left_leg": (255, 0, 0),
        "right_leg": (255, 0, 255),
        "head": (255, 255, 255),
        "spine": (128, 0, 255),
        "chest": (255, 255, 0),
    }
    for start, end in TRUNK_LIMB_CONNECTIONS:
        p1 = pt(start)
        p2 = pt(end)
        if p1 and p2:
            if (start, end) in [(11,12), (11,23), (12,24), (23,24)]:
                color = SKELETON_COLORS["torso"]
            elif start in (11,13,15) or end in (11,13,15):
                color = SKELETON_COLORS["left_arm"]
            elif start in (12,14,16) or end in (12,14,16):
                color = SKELETON_COLORS["right_arm"]
            elif start in (23,25,27) or end in (23,25,27):
                color = SKELETON_COLORS["left_leg"]
            elif start in (24,26,28) or end in (24,26,28):
                color = SKELETON_COLORS["right_leg"]
            else:
                color = (200, 200, 200)
            cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    # 脊柱与胸腔
    left_shoulder = pt(11)
    right_shoulder = pt(12)
    left_hip = pt(23)
    right_hip = pt(24)
    if left_shoulder and right_shoulder and left_hip and right_hip:
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                        (left_shoulder[1] + right_shoulder[1]) // 2)
        mid_hip = ((left_hip[0] + right_hip[0]) // 2,
                   (left_hip[1] + right_hip[1]) // 2)
        cv2.line(image, mid_shoulder, mid_hip, SKELETON_COLORS["spine"], thickness=3, lineType=cv2.LINE_AA)
        cv2.line(image, left_shoulder, right_shoulder, SKELETON_COLORS["chest"], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(image, left_hip, right_hip, SKELETON_COLORS["chest"], thickness=2, lineType=cv2.LINE_AA)

    for i in range(11, 33):
        p = pt(i)
        if p:
            cv2.circle(image, p, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    left_ear = pt(7)
    right_ear = pt(8)
    nose = pt(0)
    if left_ear and right_ear:
        head_center = ((left_ear[0] + right_ear[0]) // 2, (left_ear[1] + right_ear[1]) // 2)
        radius = int(np.linalg.norm(np.array(left_ear) - np.array(right_ear)) * 0.6)
        if radius > 0:
            cv2.ellipse(image, head_center, (radius, radius), 0, 0, 360,
                        SKELETON_COLORS["head"], 2, lineType=cv2.LINE_AA)
    elif nose:
        cv2.circle(image, nose, 8, SKELETON_COLORS["head"], 2, lineType=cv2.LINE_AA)

# ---------- 局部放大辅助函数 ----------
def _get_hand_regions(landmarks, w, h):
    regions = []
    for wrist_idx, elbow_idx in [(15, 13), (16, 14)]:
        if wrist_idx < len(landmarks) and landmarks[wrist_idx] is not None:
            x_c = int(landmarks[wrist_idx][0] * w)
            y_c = int(landmarks[wrist_idx][1] * h)
            size = 60
            if elbow_idx < len(landmarks) and landmarks[elbow_idx] is not None:
                ex = int(landmarks[elbow_idx][0] * w)
                ey = int(landmarks[elbow_idx][1] * h)
                size = int(np.linalg.norm((x_c - ex, y_c - ey)) * 0.8)
            size = max(30, min(size, 150))
            x1, y1 = max(0, x_c - size), max(0, y_c - size)
            x2, y2 = min(w, x_c + size), min(h, y_c + size)
            regions.append((x1, y1, x2, y2))
        else:
            regions.append(None)
    return regions

def _get_foot_regions(landmarks, w, h):
    regions = []
    for ankle_idx, knee_idx in [(27, 25), (28, 26)]:
        if ankle_idx < len(landmarks) and landmarks[ankle_idx] is not None:
            x_c = int(landmarks[ankle_idx][0] * w)
            y_c = int(landmarks[ankle_idx][1] * h)
            size = 50
            if knee_idx < len(landmarks) and landmarks[knee_idx] is not None:
                kx = int(landmarks[knee_idx][0] * w)
                ky = int(landmarks[knee_idx][1] * h)
                size = int(np.linalg.norm((x_c - kx, y_c - ky)) * 0.6)
            size = max(25, min(size, 120))
            x1, y1 = max(0, x_c - size), max(0, y_c - size)
            x2, y2 = min(w, x_c + size), min(h, y_c + size)
            regions.append((x1, y1, x2, y2))
        else:
            regions.append(None)
    return regions

def _overlay_zoom(main_frame, region, original_frame):
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return main_frame
    crop = original_frame[y1:y2, x1:x2].copy()
    zoom_w, zoom_h = 150, 150
    zoomed = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(zoomed, (0, 0), (zoom_w - 1, zoom_h - 1), (255, 255, 255), 2)
    margin = 10
    x_offset = main_frame.shape[1] - zoom_w - margin
    y_offset = main_frame.shape[0] - zoom_h - margin
    if x_offset < 0: x_offset = 0
    if y_offset < 0: y_offset = 0
    main_frame[y_offset:y_offset + zoom_h, x_offset:x_offset + zoom_w] = zoomed
    return main_frame

# ---------- 渲染骨架视频（含放大功能） ----------
def render_skeleton_video(video_path, skeletons, output_path, mirror=False, fps=None,
                          draw_on_black=False, detail_zone=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = orig_fps

    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        orig_frames.append(frame)
    cap.release()

    if not orig_frames:
        return None

    h, w = orig_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = min(len(orig_frames), len(skeletons))
    for i in range(total_frames):
        frame = orig_frames[i].copy()
        if draw_on_black:
            frame = np.zeros_like(frame)
        sk = skeletons[i]
        if sk is not None:
            draw_skeleton_pretty(frame, sk)
            if detail_zone == 'hands':
                regions = _get_hand_regions(sk, w, h)
                for region in regions:
                    if region is not None:
                        frame = _overlay_zoom(frame, region, orig_frames[i])
            elif detail_zone == 'feet':
                regions = _get_foot_regions(sk, w, h)
                for region in regions:
                    if region is not None:
                        frame = _overlay_zoom(frame, region, orig_frames[i])
        writer.write(frame)
    writer.release()
    return output_path

# ---------- DTW 对齐 ----------
def dtw_alignment(ref_seq, user_seq):
    if not ref_seq or not user_seq:
        return list(range(len(user_seq)))
    def get_hip_center(landmarks):
        if landmarks is None:
            return None
        lh = np.array(landmarks[23][:2])
        rh = np.array(landmarks[24][:2])
        return tuple((lh + rh) / 2)
    ref_centers = [get_hip_center(sk) if sk else None for sk in ref_seq]
    user_centers = [get_hip_center(sk) if sk else None for sk in user_seq]
    valid_ref = [i for i, c in enumerate(ref_centers) if c is not None]
    valid_user = [i for i, c in enumerate(user_centers) if c is not None]
    if not valid_ref or not valid_user:
        return list(range(len(user_seq)))
    ref_points = np.array([ref_centers[i] for i in valid_ref])
    user_points = np.array([user_centers[i] for i in valid_user])
    from scipy.spatial.distance import cdist
    dist = cdist(ref_points, user_points, metric='euclidean')
    n, m = dist.shape
    DTW = np.full((n+1, m+1), np.inf)
    DTW[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist[i-1, j-1]
            DTW[i, j] = cost + min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            prev = min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
            if prev == DTW[i-1, j-1]:
                i -= 1; j -= 1
            elif prev == DTW[i-1, j]:
                i -= 1
            else:
                j -= 1
    path.reverse()
    ref_index_map = [valid_ref[p[0]] for p in path]
    user_index_map = [valid_user[p[1]] for p in path]
    full_map = [0] * len(user_seq)
    for ui, ri in zip(user_index_map, ref_index_map):
        full_map[ui] = ri
    last_ri = 0
    for i in range(len(full_map)):
        if full_map[i] == 0 and user_seq[i] is not None:
            full_map[i] = last_ri
        elif full_map[i] != 0:
            last_ri = full_map[i]
    return full_map

# ---------- 人性化评分函数 ----------
def calculate_score_human(ref_skeletons, user_skeletons, alignment_map, weights=None):
    """基于关节点距离的姿态相似度评分，返回总分、分项得分、评语列表"""
    if weights is None:
        weights = {name: 1.0 for name in JOINT_GROUPS}

    group_distances = {name: [] for name in JOINT_GROUPS}
    total_frames = 0
    for i, user_sk in enumerate(user_skeletons):
        if i >= len(alignment_map):
            break
        ref_idx = alignment_map[i]
        if ref_idx >= len(ref_skeletons):
            continue
        ref_sk = ref_skeletons[ref_idx]
        if ref_sk is None or user_sk is None:
            continue
        total_frames += 1
        for group_name, indices in JOINT_GROUPS.items():
            dist_sum = 0.0
            count = 0
            for idx in indices:
                r = np.array(ref_sk[idx][:2])
                u = np.array(user_sk[idx][:2])
                dist_sum += np.linalg.norm(r - u)
                count += 1
            if count > 0:
                group_distances[group_name].append(dist_sum / count)

    if total_frames == 0:
        return 50, {}, ["无法评估，请重新录制。"]

    component_scores = {}
    max_possible_dist = 0.5
    for name, dists in group_distances.items():
        avg_dist = np.mean(dists) if dists else 0.0
        score = max(0, 100 - (avg_dist / max_possible_dist) * 100)
        component_scores[name] = round(score, 1)

    total_weight = sum(weights.get(n, 1) for n in component_scores)
    if total_weight > 0:
        total_score = sum(component_scores[n] * weights.get(n, 1) for n in component_scores) / total_weight
    else:
        total_score = 0

    feedback_lines = []
    sorted_groups = sorted(component_scores.items(), key=lambda x: x[1])
    worst_groups = [g for g, s in sorted_groups if s < 60]
    best_groups = [g for g, s in sorted_groups if s > 80]

    if total_score >= 80:
        feedback_lines.append("🌟 整体表现非常棒！你的动作已经很接近原版了。")
        if best_groups:
            feedback_lines.append(f"特别是{'、'.join(best_groups)}部分，做得相当到位。")
    elif total_score >= 60:
        feedback_lines.append("👍 整体还不错哦，继续努力！")
        if worst_groups:
            feedback_lines.append(f"建议多关注{'、'.join(worst_groups)}的动作细节。")
    else:
        feedback_lines.append("💪 加油！还有很大的提升空间。")
        if worst_groups:
            feedback_lines.append(f"尤其要注意{'、'.join(worst_groups)}，可以放慢速度仔细模仿。")

    for g, s in sorted_groups:
        if s < 40:
            if "手臂" in g:
                feedback_lines.append(f"🔄 {g}的动作幅度需要调整，注意肘关节的位置。")
            elif "腿" in g:
                feedback_lines.append(f"🦵 {g}的移动轨迹与原版偏差较大，注意膝盖的弯曲和落脚点。")
            elif "躯干" in g:
                feedback_lines.append(f"🧍 身体核心（腰腹、脊柱）的线条可以更贴近原版，试试保持同样的身体角度。")
            elif "头部" in g:
                feedback_lines.append(f"👤 头部位置与原视频有差异，注意跟随身体的律动。")

    return round(total_score, 1), component_scores, feedback_lines


def generate_comparison_video(ref_video_path, user_video_path, ref_skeletons, user_skeletons,
                              alignment_map, output_path, view_mode="side_by_side", mirror_ref=False):
    cap_ref = cv2.VideoCapture(ref_video_path)
    cap_user = cv2.VideoCapture(user_video_path)
    ref_fps = cap_ref.get(cv2.CAP_PROP_FPS)
    user_fps = cap_user.get(cv2.CAP_PROP_FPS)
    out_fps = min(ref_fps, user_fps)
    ref_frames = []
    while True:
        ret, frame = cap_ref.read()
        if not ret: break
        if mirror_ref: frame = cv2.flip(frame, 1)
        ref_frames.append(frame)
    user_frames = []
    while True:
        ret, frame = cap_user.read()
        if not ret: break
        user_frames.append(frame)
    cap_ref.release(); cap_user.release()
    aligned_pairs = []
    for ui in range(len(user_frames)):
        if ui < len(alignment_map):
            ri = alignment_map[ui]
            if ri < len(ref_frames):
                aligned_pairs.append((ri, ui))
    if not aligned_pairs: return None
    sample_frame = ref_frames[0]
    if view_mode == "side_by_side":
        h, w = sample_frame.shape[:2]; out_w = w*2; out_h = h
    else:
        out_w = sample_frame.shape[1]; out_h = sample_frame.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
    for ri, ui in aligned_pairs:
        ref_frame = ref_frames[ri].copy()
        user_frame = user_frames[ui].copy()
        if ri < len(ref_skeletons) and ref_skeletons[ri] is not None:
            draw_skeleton_pretty(ref_frame, ref_skeletons[ri])
        if ui < len(user_skeletons) and user_skeletons[ui] is not None:
            draw_skeleton_pretty(user_frame, user_skeletons[ui])
        if view_mode == "side_by_side":
            combined = np.hstack((ref_frame, user_frame))
        else:
            overlay = user_frame.copy()
            if ri < len(ref_skeletons) and ref_skeletons[ri] is not None:
                draw_skeleton_pretty(overlay, ref_skeletons[ri])
            combined = cv2.addWeighted(user_frame, 0.5, overlay, 0.5, 0)
        writer.write(combined)
    writer.release()
    return output_path

def extract_keyframes(skeletons, top_n=5):
    if not skeletons: return []
    movement = []
    for i in range(1, len(skeletons)):
        if skeletons[i-1] is not None and skeletons[i] is not None:
            diff = sum(np.linalg.norm(np.array(skeletons[i][j][:2]) - np.array(skeletons[i-1][j][:2])) for j in range(33))
            movement.append(diff)
        else:
            movement.append(0)
    if not movement: return []
    local_max = argrelextrema(np.array(movement), np.greater)[0]
    top_indices = sorted(local_max, key=lambda x: movement[x], reverse=True)[:top_n]
    return [idx + 1 for idx in top_indices]

def get_thumbnail(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((200, 200))
        return img
    return None

# ------------------------- WebRTC 录制处理 -------------------------
class DanceVideoProcessor(VideoProcessorBase):
    def __init__(self, recording_event, video_writer_container):
        self.recording_event = recording_event
        self.video_writer_container = video_writer_container
        self.pose = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
            draw_skeleton_pretty(img, landmarks)
        if self.recording_event.is_set():
            writer = self.video_writer_container.get("writer")
            if writer is not None:
                writer.write(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self.pose.close()

# ------------------------- Streamlit 界面 -------------------------
def main():
    st.title("🕺 舞迹 - AI舞蹈扒舞与成长记录")
    st.markdown("上传参考视频、录制或上传你的舞蹈，获得实时对比与教练级反馈。**所有数据仅在本地处理。**")

    menu = st.sidebar.selectbox("选择功能", [
        "1️⃣ 导入参考视频",
        "2️⃣ 录制你的舞蹈",
        "3️⃣ 对比与分析",
        "4️⃣ 训练历史"
    ])

    # 初始化 session_state 变量
    if "ref_video_path" not in st.session_state:
        st.session_state.ref_video_path = None
    if "ref_skeletons" not in st.session_state:
        st.session_state.ref_skeletons = None
    if "ref_fps" not in st.session_state:
        st.session_state.ref_fps = 25
    if "user_video_path" not in st.session_state:
        st.session_state.user_video_path = None
    if "user_skeletons" not in st.session_state:
        st.session_state.user_skeletons = None
    if "recording_event" not in st.session_state:
        st.session_state.recording_event = threading.Event()
    if "writer_container" not in st.session_state:
        st.session_state.writer_container = {"writer": None}
    if "history" not in st.session_state:
        with open(HISTORY_FILE, "r") as f:
            st.session_state.history = json.load(f)
    if "ref_skeleton_bg_video" not in st.session_state:
        st.session_state.ref_skeleton_bg_video = None
    if "ref_skeleton_black_video" not in st.session_state:
        st.session_state.ref_skeleton_black_video = None
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []
    if "loop_segments" not in st.session_state:
        st.session_state.loop_segments = []
    if "jump_to_time" not in st.session_state:
        st.session_state.jump_to_time = None

    # ======================== 1. 导入参考视频 ========================
    if menu == "1️⃣ 导入参考视频":
        st.header("导入参考视频")
        uploaded_file = st.file_uploader("上传视频文件(MP4/MOV/AVI)", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            temp_ref_path = VIDEO_DIR / "reference_video.mp4"
            with open(temp_ref_path, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state.ref_video_path = str(temp_ref_path)

        col1, col2 = st.columns(2)
        mirror_ref = col1.checkbox("镜面翻转(方便跟跳)", value=False)
        slow_factor = col2.selectbox("慢放速率", [1.0, 0.75, 0.5], index=0)
        detail_zoom = st.selectbox("细节放大", ["无", "手部", "脚步"], index=0,
                                   help="在视频右下角显示指定部位的真实画面放大视图")

        if st.session_state.ref_video_path and os.path.exists(st.session_state.ref_video_path):
            if st.button("开始提取骨架"):
                st.cache_data.clear()
                skeletons, fps = extract_skeleton_from_video(
                    st.session_state.ref_video_path, fps=None, mirror=mirror_ref)
                st.session_state.ref_skeletons = skeletons
                st.session_state.ref_fps = fps

                output_fps = fps
                if slow_factor != 1.0:
                    output_fps = fps * slow_factor

                zone_map = {"手部": "hands", "脚步": "feet"}
                detail_zone = zone_map.get(detail_zoom, None)

                skeleton_bg_path = str(VIDEO_DIR / "reference_skeleton_bg.mp4")
                render_skeleton_video(st.session_state.ref_video_path, skeletons, skeleton_bg_path,
                                      mirror=mirror_ref, fps=output_fps, draw_on_black=False,
                                      detail_zone=detail_zone)
                st.session_state.ref_skeleton_bg_video = skeleton_bg_path

                skeleton_black_path = str(VIDEO_DIR / "reference_skeleton_black.mp4")
                render_skeleton_video(st.session_state.ref_video_path, skeletons, skeleton_black_path,
                                      mirror=mirror_ref, fps=output_fps, draw_on_black=True,
                                      detail_zone=detail_zone)
                st.session_state.ref_skeleton_black_video = skeleton_black_path

                np.save(SKELETON_DIR / "ref_skeleton.npy", np.array(skeletons, dtype=object))
                st.success(f"骨架提取完成，共 {len(skeletons)} 帧，输出帧率 {output_fps:.1f}")

            if st.session_state.ref_skeletons is not None:
                video_option = st.radio("显示模式", ["带背景骨架", "纯黑背景骨架"], index=0)
                vid_path = st.session_state.ref_skeleton_bg_video if video_option == "带背景骨架" else st.session_state.ref_skeleton_black_video

                if vid_path and os.path.exists(vid_path):
                    col_left, col_video, col_right = st.columns([2, 1, 2])
                    with col_video:
                        play_start = 0.0
                        if st.session_state.jump_to_time is not None:
                            play_start = st.session_state.jump_to_time
                            st.session_state.jump_to_time = None

    # 直接使用 st.video 的 start_time 参数（需要 Streamlit >= 1.29）
                        st.video(vid_path, start_time=int(play_start))  

                    # ---- 标记与循环段 ----
                    tab1, tab2 = st.tabs(["📍 单点标记", "🔁 循环段标记"])

                    with tab1:
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            mark_time = st.number_input("时间 (秒)", min_value=0.0, step=1.0, key="mark_time")
                        with col_b:
                            mark_label = st.text_input("名称", placeholder="例：副歌前八拍", key="mark_label")
                        if st.button("➕ 添加标记"):
                            st.session_state.bookmarks.append((mark_time, mark_label))
                            st.success(f"已标记 {mark_time:.1f}s")
                            st.rerun()
                        if st.session_state.bookmarks:
                            for idx, (t, label) in enumerate(st.session_state.bookmarks):
                                cols = st.columns([1, 3, 1, 1])
                                with cols[0]: st.code(f"{t:.1f}s")
                                with cols[1]: st.write(f"**{label}**")
                                with cols[2]:
                                    if st.button("▶️ 跳转", key=f"jump_{idx}"):
                                        st.session_state.jump_to_time = t
                                        st.rerun()
                                with cols[3]:
                                    if st.button("✕", key=f"del_bm_{idx}"):
                                        st.session_state.bookmarks.pop(idx)
                                        st.rerun()
                            if st.button("🗑️ 清除所有标记"):
                                st.session_state.bookmarks.clear()
                                st.rerun()

                    with tab2:
                        st.caption("设定一个时间段，可反复循环播放，方便跟练。")
                        col_c, col_d = st.columns(2)
                        with col_c:
                            loop_start = st.number_input("开始 (秒)", min_value=0.0, step=1.0, key="loop_start")
                        with col_d:
                            loop_end = st.number_input("结束 (秒)", min_value=0.0, step=1.0, key="loop_end")
                        loop_label = st.text_input("段名称", key="loop_label", placeholder="例：副歌部分")
                        if st.button("➕ 添加循环段"):
                            if loop_end > loop_start:
                                st.session_state.loop_segments.append((loop_start, loop_end, loop_label))
                                st.success(f"循环段已添加：{loop_start}-{loop_end}s")
                                st.rerun()
                            else:
                                st.error("结束时间必须大于开始时间")
                        if st.session_state.loop_segments:
                            for idx, (s, e, lbl) in enumerate(st.session_state.loop_segments):
                                cols = st.columns([1, 1, 2, 1, 1])
                                with cols[0]: st.code(f"{s:.1f}s")
                                with cols[1]: st.code(f"{e:.1f}s")
                                with cols[2]: st.write(f"**{lbl}**")
                                with cols[3]:
                                    if st.button("🔁 循环", key=f"loop_{idx}"):
                                        st.session_state.loop_playback = (vid_path, s, e)
                                        st.rerun()
                                with cols[4]:
                                    if st.button("✕", key=f"del_loop_{idx}"):
                                        st.session_state.loop_segments.pop(idx)
                                        st.rerun()
                            if st.button("🗑️ 清除所有循环段"):
                                st.session_state.loop_segments.clear()
                                st.rerun()

                    # 处理循环播放弹出（同样使用动态 id）
                    if "loop_playback" in st.session_state and st.session_state.loop_playback:
                        loop_vid, loop_s, loop_e = st.session_state.loop_playback
                        loop_unique = f"loop_{int(loop_s*100)}_{int(loop_e*100)}_{int(time.time()*1000000)}"
                        st.markdown("#### 🔄 循环播放中...")
                        components.html(f"""
                        <video id="{loop_unique}" width="100%" controls autoplay>
                            <source src="{loop_vid}" type="video/mp4">
                        </video>
                        <script>
                            const video = document.getElementById('{loop_unique}');
                            video.currentTime = {loop_s};
                            video.addEventListener('timeupdate', () => {{
                                if (video.currentTime >= {loop_e}) {{
                                    video.currentTime = {loop_s};
                                    video.play();
                                }}
                            }});
                        </script>
                        """, height=400)
                        if st.button("❌ 关闭循环播放"):
                            del st.session_state.loop_playback
                            st.rerun()

        else:
            st.info("请先上传视频")

    # ======================== 2. 录制你的舞蹈 ========================
    elif menu == "2️⃣ 录制你的舞蹈":
        st.header("录制你的舞蹈")
        source = st.radio("选择来源", ["打开摄像头录制", "上传本地视频"], index=0)

        if source == "打开摄像头录制":
            st.markdown("打开摄像头，点击“开始录制”后跳舞，系统将实时显示骨架。")
            recording_event = st.session_state.recording_event
            writer_container = st.session_state.writer_container

            col1, col2 = st.columns(2)
            if col1.button("🟢 开始录制"):
                recording_event.set()
                temp_user_path = str(VIDEO_DIR / "user_dance.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                writer = cv2.VideoWriter(temp_user_path, fourcc, 20.0, (640, 480))
                writer_container["writer"] = writer
                st.session_state.user_video_path = temp_user_path
                col1.success("录制中...")

            if col2.button("🔴 停止录制"):
                recording_event.clear()
                if writer_container["writer"] is not None:
                    writer_container["writer"].release()
                    writer_container["writer"] = None
                col2.info("录制已停止，视频已保存")

            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_ctx = webrtc_streamer(
                key="dance-record",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=lambda: DanceVideoProcessor(recording_event, writer_container),
                media_stream_constraints={"video": True, "audio": True},
                async_processing=True,
            )

            if st.session_state.user_video_path and os.path.exists(st.session_state.user_video_path):
                st.video(st.session_state.user_video_path)

        else:
            user_upload = st.file_uploader("选择已录制的舞蹈视频", type=["mp4", "mov", "avi"], key="user_upload")
            if user_upload is not None:
                user_vid_path = str(VIDEO_DIR / "user_dance.mp4")
                with open(user_vid_path, "wb") as f:
                    f.write(user_upload.read())
                st.session_state.user_video_path = user_vid_path
                st.success("视频上传成功！可以前往“对比与分析”进行评测。")
                st.video(user_vid_path)

    # ======================== 3. 对比与分析 ========================
    elif menu == "3️⃣ 对比与分析":
        st.header("对比与分析")
        if not st.session_state.ref_video_path or not os.path.exists(st.session_state.ref_video_path):
            st.warning("请先导入参考视频(步骤1)"); return
        if not st.session_state.user_video_path or not os.path.exists(st.session_state.user_video_path):
            st.warning("请先录制或上传你的舞蹈(步骤2)"); return

        if st.session_state.user_skeletons is None and st.session_state.user_video_path:
            user_vid = st.session_state.user_video_path
            if not os.path.exists(user_vid) or os.path.getsize(user_vid) == 0:
                st.error("用户视频文件不存在或为空，请重新录制/上传。"); return
            cap_test = cv2.VideoCapture(user_vid)
            if not cap_test.isOpened():
                st.error("无法读取用户视频，可能已损坏。"); cap_test.release(); return
            cap_test.release()
            with st.spinner("正在提取用户骨架..."):
                try:
                    user_skels, user_fps = extract_skeleton_from_video(user_vid, fps=None)
                    st.session_state.user_skeletons = user_skels
                    np.save(SKELETON_DIR / "user_skeleton.npy", np.array(user_skels, dtype=object))
                    st.success("用户骨架提取完成")
                except Exception as e:
                    st.error(f"骨架提取失败：{e}"); return

        ref_skels = st.session_state.ref_skeletons
        user_skels = st.session_state.user_skeletons
        if ref_skels is None or user_skels is None:
            st.error("骨架数据缺失，请返回前两步处理。"); return

        st.subheader("时间对齐")
        align_method = st.radio("选择对齐方式", ["手动偏移(滑块)", "自动DTW对齐"], index=0)
        if align_method == "手动偏移(滑块)":
            max_offset = max(0, len(user_skels) - 1)
            offset = st.slider("用户视频起始帧偏移", 0, max_offset, 0)
            alignment_map = [min(i + offset, len(ref_skels) - 1) for i in range(len(user_skels))]
        else:
            if st.button("执行DTW自动对齐"):
                with st.spinner("DTW计算中..."):
                    alignment_map = dtw_alignment(ref_skels, user_skels)
                st.session_state.alignment_map = alignment_map
                st.success("自动对齐完成")
            alignment_map = st.session_state.get("alignment_map", list(range(min(len(ref_skels), len(user_skels)))))

        st.subheader("评分与反馈")
        weights = {name: 1.0 for name in JOINT_GROUPS}
        with st.expander("自定义评分权重"):
            cols = st.columns(3)
            for i, (name, _) in enumerate(JOINT_GROUPS.items()):
                with cols[i % 3]:
                    weights[name] = st.slider(f"{name}", 0.0, 2.0, 1.0, 0.1, key=f"w_{name}")

        total_score, component_scores, feedback_lines = calculate_score_human(
            ref_skels, user_skels, alignment_map, weights)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("整体相似度", f"{total_score} / 100")
            st.caption("分数越高，动作越接近原版")
            st.subheader("各部位得分")
            for name, score in component_scores.items():
                st.write(f"{name}: {score}")
        with col2:
            st.subheader("教练点评")
            for line in feedback_lines:
                st.write(line)

        # 用于历史记录的偏差值
        avg_deviation = {}
        for name in JOINT_GROUPS:
            avg_deviation[name] = max(0, 100 - component_scores.get(name, 100))

        st.subheader("对比视图")
        view_mode = st.radio("视图模式", ["并排对比", "半透明叠加"], index=0)
        mode_key = "side_by_side" if view_mode == "并排对比" else "overlay"
        if st.button("生成对比视频"):
            with st.spinner("渲染中..."):
                output_path = str(VIDEO_DIR / "comparison.mp4")
                generate_comparison_video(
                    st.session_state.ref_video_path, st.session_state.user_video_path,
                    ref_skels, user_skels, alignment_map, output_path,
                    view_mode=mode_key, mirror_ref=st.session_state.get("mirror_ref", False))
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button("下载对比视频", f, file_name="dance_comparison.mp4")

        if st.button("保存本次成绩到历史记录"):
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": total_score,
                "component_scores": component_scores,
                "deviations": avg_deviation,
                "view_mode": view_mode
            }
            st.session_state.history.append(record)
            with open(HISTORY_FILE, "w") as f:
                json.dump(st.session_state.history, f, indent=2)
            st.success("已保存")

    # ======================== 4. 训练历史 ========================
    elif menu == "4️⃣ 训练历史":
        st.header("训练历史与趋势")
        if not st.session_state.history:
            st.info("暂无历史记录，请先完成一次对比并保存。")
        else:
            dates = [r["timestamp"] for r in st.session_state.history]
            scores = [r["total_score"] for r in st.session_state.history]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=scores, mode='lines+markers', name='相似度'))
            fig.update_layout(title="成长趋势", xaxis_title="日期", yaxis_title="相似度")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("详细记录")
            for record in reversed(st.session_state.history):
                with st.expander(f"{record['timestamp']} - 相似度: {record['total_score']}"):
                    st.json(record)

            if st.button("导出最近一次训练报告"):
                if st.session_state.history:
                    last = st.session_state.history[-1]
                    report = f"训练时间：{last['timestamp']}\n整体相似度：{last['total_score']}/100\n各部位得分：\n"
                    for name, score in last['component_scores'].items():
                        report += f"- {name}: {score}\n"
                    st.download_button("下载训练报告", report, file_name="dance_report.txt")

if __name__ == "__main__":
    main()