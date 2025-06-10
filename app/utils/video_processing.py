"""
utils/video_processing.py
=========================
• process_uploaded_video  – офф-лайн: файл → файл
• process_video_realtime  – ⚡ кадр-за-кадром для Streamlit
• process_rtsp_stream     – live-камера / RTSP
"""

from __future__ import annotations
import cv2
import os
import tempfile
import logging
import time
from typing import Dict, Iterator, Tuple, Optional

from utils.image_processing import draw_boxes
from utils.database import log_detections_bulk
from utils.chat_comment import summarize_detections

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
def process_uploaded_video(
    video_path: str,
    model,
    conf_thresh: float,
    line_thickness: int,
    db_session,
) -> str:
    """
    Полностью обрабатывает видеофайл, сохраняет размеченный .mp4 и
    возвращает путь к нему.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = model.predict(frame, conf=conf_thresh)
        frame = draw_boxes(frame, detections, conf_thresh, model, line_thickness)
        writer.write(frame)

    cap.release()
    writer.release()
    logger.info("Видео обработано: %s → %s", video_path, out_path)
    return out_path


# ───────────────────────────────────────────────────────────────────────────────
def process_video_realtime(
    video_path: str,
    model,
    conf_thresh: float,
    line_thickness: int,
    db_session,
    gpt_every: int = 60,
) -> Iterator[Tuple]:
    """
    Генератор: даёт (BGR-кадр, Optional[str-gptComment]) каждую итерацию.
    Используется во вкладке «⚡ Видео-файл online».
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    frame_idx: int = 0
    counts: Dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = model.predict(frame, conf=conf_thresh)
        frame = draw_boxes(frame, detections, conf_thresh, model, line_thickness)

        # tally
        for det in detections:
            cls_id = int(det[5])
            name = model.model.names[cls_id]
            counts[name] = counts.get(name, 0) + 1

        # пишем в БД раз в 30 кадров
        if frame_idx % 30 == 0:
            try:
                log_detections_bulk(db_session, detections)
            except Exception:
                pass

        gpt_comment: Optional[str] = None
        if frame_idx and frame_idx % gpt_every == 0:
            gpt_comment = summarize_detections(counts)
            counts = {}

        frame_idx += 1
        yield frame, gpt_comment

    cap.release()


# ───────────────────────────────────────────────────────────────────────────────
def process_rtsp_stream(
    rtsp_url: str,
    model,
    conf_thresh: float,
    line_thickness: int,
    db_session,
) -> None:
    """
    Live-поток RTSP / IP-камера. Показывает кадры в Streamlit.
    """
    import streamlit as st

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось подключиться к RTSP: {rtsp_url}")

    st_frame = st.empty()
    st_comment = st.empty()
    frame_idx: int = 0
    counts: Dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = model.predict(frame, conf=conf_thresh)
        frame = draw_boxes(frame, detections, conf_thresh, model, line_thickness)

        # счёт
        for det in detections:
            cls_id = int(det[5])
            name = model.model.names[cls_id]
            counts[name] = counts.get(name, 0) + 1

        # вывод кадра
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_column_width=True)

        # GPT-комментарий каждые 60 кадров
        if frame_idx and frame_idx % 60 == 0:
            st_comment.markdown(
                f"**💬 GPT:** {summarize_detections(counts)}")
            counts = {}

        # БД
        if frame_idx % 30 == 0:
            try:
                log_detections_bulk(db_session, detections)
            except Exception:
                pass

        frame_idx += 1

    cap.release()
