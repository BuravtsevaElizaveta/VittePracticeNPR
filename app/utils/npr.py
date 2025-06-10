# utils/npr.py
# ======================================================================
#  Распознавание номерного знака, марки, цвета, типа ТС и расширенное
#  GPT-описание автомобиля.  Совместимо с Python 3.9.
# ======================================================================

import base64
import json
import logging
import time
from functools import lru_cache
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf

from openai_config import get_client

logger = logging.getLogger(__name__)
_oai = get_client()

# ─────────────────────────── пути к ресурсам ──────────────────────────
MODEL_H5_PATH = "models/model.h5"
CASCADE_PATH = "models/cascades/haarcascade_licence_plate_rus_16stages.xml"

# ─────────────────────────── алфавит CNN ──────────────────────────────
_CHARS = "0123456789АВЕКМНОРСТУХ"
_NUM2CHAR = {i: c for i, c in enumerate(_CHARS)}

# ─────────────────────────── ленивые ресурсы ──────────────────────────
@lru_cache
def _cascade():
    return cv2.CascadeClassifier(CASCADE_PATH)


@lru_cache
def _cnn():
    return tf.keras.models.load_model(MODEL_H5_PATH)


# ─────────────────────────── общие GPT-вызовы ─────────────────────────
def _img_b64(img_bgr: np.ndarray, max_size: int = 224, quality: int = 40) -> str:
    h, w = img_bgr.shape[:2]
    scale = max_size / float(max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(enc).decode() if ok else ""


def _chat_gpt(system_prompt: str, user_prompt: str, img_b64: str = "", retries: int = 5) -> str:
    """
    Унифицированный вызов ChatCompletion Vision.  Возвращает content-строку.
    """
    for i in range(retries):
        try:
            msg = [{"role": "system", "content": system_prompt},
                   {"role": "user",
                    "content": ([{"type": "text", "text": user_prompt}] +
                                ([{"type": "image_url",
                                   "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]
                                 if img_b64 else []))}]
            resp = _oai.chat.completions.create(model="gpt-4o-mini",
                                                messages=msg,
                                                temperature=0.0,
                                                max_tokens=512)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("GPT (%s/%s) error: %s", i + 1, retries, e)
            time.sleep(1 + i)
    return ""


# ─────────────────────────── Haar → ROI ───────────────────────────────
def detect_plate(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    plate_img = img_bgr.copy()
    rects = _cascade().detectMultiScale(plate_img, 1.1, 5)
    roi = None
    for (x, y, w, h) in rects:
        roi = plate_img[y:y + h, x:x + w]
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return plate_img, roi


# ─────────────────────────── сегментация ──────────────────────────────
def _segment_characters(roi: np.ndarray) -> List[np.ndarray]:
    if roi is None:
        return []
    lp = cv2.resize(roi, (333, 75))
    gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    thr[:3, :], thr[-3:, :], thr[:, :3], thr[:, -3:] = 255, 255, 255, 255

    H, W = thr.shape
    min_h, max_h = H / 6, H / 2
    min_w, max_w = W / 10, 2 * W / 3

    cntrs, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        if min_w < w < max_w and min_h < h < max_h:
            ch = thr[y:y + h, x:x + w]
            chars.append((x, ch))
    chars = [ch for x, ch in sorted(chars, key=lambda t: t[0])]
    return [cv2.resize(ch, (28, 28)) for ch in chars]


# ─────────────────────────── CNN распознавание ────────────────────────
def recognize_plate(roi: np.ndarray, conf_thr: float = 0.15) -> Tuple[str, List[float]]:
    if roi is None:
        return "", []
    chars = _segment_characters(roi)
    if not chars:
        return "", []
    res, confs = [], []
    for ch in chars:
        img = np.stack((ch,) * 3, axis=-1) / 255.0
        pr = _cnn().predict(img.reshape(1, 28, 28, 3), verbose=0)
        idx, conf = int(np.argmax(pr)), float(np.max(pr))
        res.append(_NUM2CHAR[idx] if conf >= conf_thr else "?")
        confs.append(conf)
    return "".join(res), confs


# ─────────────────────────── GPT-fallback номер ───────────────────────
def recognize_plate_gpt(img_bgr: np.ndarray) -> str:
    prompt = ("На изображении виден государственный номер автомобиля. "
              "Считай символы точно и верни **только** номер без лишних слов. "
              "Если номер не различим, напиши UNKNOWN.")
    return _chat_gpt("Ты OCR-ассистент.", prompt, _img_b64(img_bgr))


# ─────────────────────────── марка / цвет / тип ───────────────────────
def detect_brand(img_bgr: np.ndarray) -> str:
    return _chat_gpt("Ты определяешь марку авто по фото.",
                     "Назови только марку автомобиля:",
                     _img_b64(img_bgr))


def detect_color(img_bgr: np.ndarray) -> str:
    return _chat_gpt("Определи основной цвет кузова.",
                     "Ответь одним словом (на русском):",
                     _img_b64(img_bgr))


def detect_car_type(img_bgr: np.ndarray) -> str:
    return _chat_gpt(
        "Классифицируй транспорт.",
        "Ответ: легковой, грузовой, автобус или мотоцикл (одно слово).",
        _img_b64(img_bgr),
    )


# ─────────────────────────── анализ РФ номера ─────────────────────────
def analyze_russian_number(plate: str) -> Tuple[str, str]:
    if not plate:
        return "", ""
    j = _chat_gpt(
        "Анализ российского номера.",
        ("Формат A123BC77. Выведи JSON с ключами region_name и year_issued "
         f"по номеру: {plate}"),
    )
    try:
        data = json.loads(j)
        return data.get("region_name", ""), data.get("year_issued", "")
    except Exception:
        return "", ""


# ─────────────────────────── подробное описание ───────────────────────
def describe_car(img_bgr: np.ndarray) -> str:
    return _chat_gpt(
        "Опиши автомобиль детально для автолюбителя.",
        ("Укажи предполагаемую марку, модель, примерный год выпуска, "
         "класс, интересные детали.  Не упоминай, что ты ИИ."),
        _img_b64(img_bgr, max_size=2000, quality=50),
    )
