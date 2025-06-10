# -*- coding: utf-8 -*-
"""
AI‑Автоанализатор (Streamlit)
Обновлённая версия с поддержкой YOLOv8 для обнаружения номерных знаков.

  * YOLO‑модель: utils/YOLOv8.pt
  * Дополнительные модули: директория utils (автоматически добавляется в sys.path)

Основные изменения:
  1. Загружается YOLOv8 через ultralytics.
  2. Добавлен выбор метода обнаружения (YOLOv8 | Haar Cascade) в сайдбаре.
  3. plate_detect теперь использует YOLOv8 по умолчанию с резервным переходом на
     Haar‑каскад, если YOLO не найден или не обнаружил номер.
  4. Старый plate_detect переименован в plate_detect_haar для совместимости.
"""

###############################################################################
# ИМПОРТЫ
###############################################################################
from streamlit.web import cli as stcli
import sys
from streamlit import runtime

import os
import time
import logging
import warnings
import json
import re
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import cv2
from ultralytics import YOLO   # <‑‑‑ новое
import tensorflow as tf
from PIL import Image
import openai
import concurrent.futures

###############################################################################
# КОНФИГУРАЦИЯ ПУТЕЙ / МОДЕЛЕЙ
###############################################################################
# Папка utils (с YOLO‑моделью и вспомогательными модулями)
UTILS_DIR = Path(__file__).parent / "utils"
if UTILS_DIR.exists() and str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

# Пути к весам моделей
YOLO_MODEL_PATH = UTILS_DIR / "YOLOv8.pt"
CNN_MODEL_PATH = Path("model.h5")
CASCADE_PATH = Path("haarcascade_licence_plate_rus_16stages.xml")

# Разрешённые символы для модели CNN (пример)
characters = '0123456789АВЕКМНОРСТУХ'
num_to_char = {i: ch for i, ch in enumerate(characters)}

###############################################################################
# УСТАНОВКИ STREAMLIT
###############################################################################
# Подавляем служебные предупреждения
warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext"
)
warnings.filterwarnings(
    "ignore",
    message="Session state does not function when running a script without `streamlit run`"
)
logging.getLogger('streamlit.runtime.scriptrunner_utils').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

###############################################################################
# ЛОГИРОВАНИЕ
###############################################################################
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

###############################################################################
# OPENAI (через Proxy)
###############################################################################
proxy_api_key = "sk-2uHtBOkjr3ZrCn43aUt4WdEZ20JaXu49"
proxy_base_url = "https://api.proxyapi.ru/openai/v1"
client = openai.OpenAI(api_key=proxy_api_key, base_url=proxy_base_url)

###############################################################################
# ФУНКЦИИ ДЛЯ GPT‑4o (через Proxy API)
###############################################################################

def safe_chat_completion(messages, model="gpt-4o", temperature: float = 0.0, max_retries: int = 5):
    """Запрос к GPT‑4o с повторными попытками."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                temperature=temperature,
                max_output_tokens=2000
            )
            return response
        except Exception as e:
            logging.error(f"[GPT‑4o Error]: {e}. Retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
    raise Exception("[GPT‑4o] Max retries reached in safe_chat_completion")

###############################################################################
# ЗАГРУЗКА МОДЕЛЕЙ (кэшируется Streamlit)
###############################################################################

@st.cache_resource(show_spinner="Загрузка YOLO‑модели …")
def load_yolo_model():
    if YOLO_MODEL_PATH.exists():
        try:
            model = YOLO(str(YOLO_MODEL_PATH))
            logging.info("YOLOv8 model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load YOLOv8 model: {e}")
    else:
        logging.warning("YOLOv8 weights not found – fallback to Haar cascade.")
    return None  # сигнал для фолбэка

yolo_model = load_yolo_model()


@st.cache_resource(show_spinner="Загрузка CNN‑модели …")
def load_model_cnn():
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
    model_cnn_loaded = tf.keras.models.load_model(str(CNN_MODEL_PATH))
    logging.info("CNN model loaded successfully.")
    return model_cnn_loaded

model_cnn = load_model_cnn()

###############################################################################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
###############################################################################

def recognize_brand_gpt4(img_bgr, max_size: int = 224, quality: int = 30) -> str:
    """GPT‑4o: распознаём марку автомобиля."""
    default_brand = "Не удалось распознать марку"
    h, w = img_bgr.shape[:2]
    scale = max_size / float(max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    success, encoded_img = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return default_brand

    img_base64 = base64.b64encode(encoded_img).decode()
    system_msg = "Ты — помощник, распознающий марку машины на изображении. Верни только марку без дополнительной информации."
    user_msg = [
        {"type": "input_text", "text": "Определите марку автомобиля на этом изображении."},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_base64}"}
    ]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = safe_chat_completion(messages)
        if hasattr(response, 'output') and response.output:
            return response.output[0].content[0].text.strip()
    except Exception as e:
        logging.error(f"[GPT‑4o brand] {e}")
    return default_brand


def recognize_color_gpt4(img_bgr, max_size: int = 224, quality: int = 30) -> str:
    """GPT‑4o: распознаём цвет автомобиля."""
    default_color = "Не удалось распознать цвет"
    h, w = img_bgr.shape[:2]
    scale = max_size / float(max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    success, encoded_img = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return default_color

    img_base64 = base64.b64encode(encoded_img).decode()
    system_msg = "Ты — помощник, распознающий основной цвет автомобиля на изображении."
    user_msg = [
        {"type": "input_text", "text": "Определите основной цвет автомобиля на этом изображении."},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_base64}"}
    ]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = safe_chat_completion(messages)
        if hasattr(response, 'output') and response.output:
            return response.output[0].content[0].text.strip()
    except Exception as e:
        logging.error(f"[GPT‑4o color] {e}")
    return default_color


def classify_car_type(img_bgr) -> str:
    """GPT‑4o: классифицируем тип ТС."""
    default_type = "Не удалось классифицировать тип автомобиля"
    success, encoded_img = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if not success:
        return default_type
    img_base64 = base64.b64encode(encoded_img).decode()

    system_msg = (
        "Ты — помощник, классифицирующий тип автомобиля на изображении. "
        "Классифицируй его как: легковой, грузовой, автобус, мотоцикл."
    )
    user_msg = [
        {"type": "input_text", "text": "Определите тип этого транспортного средства."},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_base64}"}
    ]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = safe_chat_completion(messages)
        if hasattr(response, 'output') and response.output:
            return response.output[0].content[0].text.strip()
    except Exception as e:
        logging.error(f"[GPT‑4o type] {e}")
    return default_type


def analyze_russian_number_gpt(plate_number: str) -> Tuple[str, str]:
    """GPT‑4o: определяем регион и год выдачи российского номера."""
    if not plate_number:
        return "", ""

    system_msg = (
        "Ты — помощник, анализирующий российский госномер. "
        "У номера формат: одна буква, три цифры, две буквы, регион (2‑3 цифры). "
        "По коду региона определи субъект РФ и предположи год выдачи. "
        "Верни JSON: {\"region_name\": \"...\", \"year_issued\": \"...\"}."
    )
    user_msg = [{"type": "input_text", "text": f"Определи регион и год выдачи для номера: {plate_number}."}]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = safe_chat_completion(messages, temperature=0.1)
        if hasattr(response, 'output') and response.output:
            try:
                data = json.loads(response.output[0].content[0].text.strip())
                return data.get("region_name", ""), data.get("year_issued", "")
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logging.error(f"[GPT‑4o region/year] {e}")
    return "", ""

###############################################################################
# ОБНАРУЖЕНИЕ НОМЕРНОГО ЗНАКА
###############################################################################

def plate_detect_haar(img: np.ndarray):
    """Старый метод: Haar Cascade."""
    plateCascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    plateImg, roi = img.copy(), img.copy()
    plate_part = None
    for (x, y, w, h) in plateCascade.detectMultiScale(plateImg, scaleFactor=1.1, minNeighbors=5):
        plate_part = roi[y:y+h, x:x+w]
        cv2.rectangle(plateImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break  # используем первый найденный
    return plateImg, plate_part


def plate_detect(img: np.ndarray, method: str = 'YOLOv8') -> Tuple[np.ndarray, np.ndarray]:
    """Объединённая функция: YOLOv8 (по умолчанию) с фолбэком на Haar."""
    if method == 'YOLOv8' and yolo_model is not None:
        try:
            results = yolo_model.predict(img, verbose=False, device='cpu')
            for r in results:
                if r.boxes:  # >= 1 объект
                    # Берём первый бокс (предполагаем «license_plate» один‑класс)
                    b = r.boxes[0].xyxy[0].cpu().numpy().astype(int).tolist()
                    x1, y1, x2, y2 = b
                    plate_part = img[y1:y2, x1:x2]
                    plateImg = img.copy()
                    cv2.rectangle(plateImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    return plateImg, plate_part
        except Exception as e:
            logging.error(f"YOLOv8 detect failed: {e}")
            # переходим к Haar

    # Fallback: Haar Cascade
    return plate_detect_haar(img)

###############################################################################
# СЕГМЕНТАЦИЯ, ПРЕ‑/ПОСТОБРАБОТКА ДЛЯ CNN
###############################################################################

def segment_characters(image: np.ndarray) -> np.ndarray:
    """Разбиваем номер на отдельные символы."""
    if image is None:
        return np.array([])
    img_lp = cv2.resize(image, (333, 75))
    img_gray = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_bin[0:3, :], img_bin[:, 0:3], img_bin[72:75, :], img_bin[:, 330:333] = 255, 255, 255, 255

    LP_W, LP_H = img_bin.shape
    dims = [LP_W / 6, LP_W / 2, LP_H / 10, 2 * LP_H / 3]
    lower_h, upper_h, lower_w, upper_w = dims[0] * 0.5, dims[1] * 1.2, dims[2] * 0.5, dims[3] * 1.2

    cntrs, hier = cv2.findContours(img_bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return np.array([])

    x_list, chars = [], []
    for i, c in enumerate(cntrs):
        if hier[0][i][3] != -1:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / h
            if lower_h < h < upper_h and lower_w < w < upper_w and 0.3 < ar < 1.2:
                x_list.append(x)
                char = cv2.subtract(255, cv2.resize(img_bin[y:y+h, x:x+w], (20, 40)))
                char_copy = np.zeros((44, 24))
                char_copy[2:42, 2:22] = char
                chars.append(char_copy)

    idx_sorted = np.argsort(x_list)
    return np.array([chars[i] for i in idx_sorted]) if chars else np.array([])


def fix_dimension(img):
    return np.stack((img,)*3, axis=-1) if img.ndim == 2 else img


def recognize_number_cnn(img_plate: np.ndarray, model, conf_thr: float = 0.0):
    seg_chars = segment_characters(img_plate)
    if seg_chars.size == 0:
        return None, []
    output, confs = [], []
    for ch in seg_chars:
        img = fix_dimension(cv2.resize(ch, (28, 28))) / 255.0
        pred = model.predict(img.reshape(1, 28, 28, 3), verbose=0)
        idx = int(np.argmax(pred))
        conf = float(pred[0, idx])
        output.append(num_to_char[idx] if conf >= conf_thr else '?')
        confs.append(conf)
    return ''.join(output), confs


def fix_number_format_new_rus(plate_number: str, confs: List[float]):
    pattern = r"^([АВЕКМНОРСТУХ])\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$"
    if re.match(pattern, plate_number):
        return plate_number, confs
    if plate_number and plate_number[-1] in "АВЕКМНОРСТУХ":
        candidate = plate_number[:-1]
        if re.match(pattern, candidate):
            return candidate, confs[:-1]
    return plate_number, confs

###############################################################################
# ЭКСПОРТ / ИМПОРТ JSON‑XML
###############################################################################

def generate_json_result(plate_number: str, brand: str, car_type: str, color: str) -> str:
    return json.dumps({
        "plate_number": plate_number or "",
        "brand": brand or "",
        "car_type": car_type or "",
        "color": color or ""
    }, ensure_ascii=False, indent=4)


def generate_xml_result(plate_number: str, brand: str, car_type: str, color: str) -> str:
    root = ET.Element("CarAnalysisResult")
    ET.SubElement(root, "PlateNumber").text = plate_number or ""
    ET.SubElement(root, "Brand").text = brand or ""
    ET.SubElement(root, "CarType").text = car_type or ""
    ET.SubElement(root, "Color").text = color or ""
    return ET.tostring(root, encoding="utf-8").decode()


def parse_json(json_str: str):
    try:
        d = json.loads(json_str)
        return d.get('plate_number', ''), d.get('brand', ''), d.get('car_type', ''), d.get('color', '')
    except Exception:
        return '', '', '', ''


def parse_xml(xml_str: str):
    try:
        r = ET.fromstring(xml_str)
        return (
            r.findtext('PlateNumber', default=''),
            r.findtext('Brand', default=''),
            r.findtext('CarType', default=''),
            r.findtext('Color', default='')
        )
    except Exception:
        return '', '', '', ''

###############################################################################
# ГЛОБАЛЬНАЯ СТАТИСТИКА
###############################################################################

def init_stats():
    if 'stat_items' not in st.session_state:
        st.session_state['stat_items'] = []


def add_stat_item(plate_number: str, region: str, year: str):
    st.session_state['stat_items'].append({
        'plate_number': plate_number,
        'region': region,
        'year_issued': year
    })
