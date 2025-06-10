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

###############################################################################
# GPT-4: АНАЛИЗ РОССИЙСКОГО НОМЕРА (регион, год)
###############################################################################
def analyze_russian_number_gpt(plate_number: str) -> (str, str):
    """
    Дополнительный функционал: GPT-4o анализирует российский номер и 
    пытается определить регион (по коду) и год выдачи (примерно).
    Возвращает (region_str, year_str).
    """
    if not plate_number:
        return ("", "")

    # Пример system- и user-сообщений
    system_msg = (
        "Ты — помощник, анализирующий российский госномер. "
        "У номера формат: одна буква, три цифры, две буквы, и регион (2 или 3 цифры), например A123BC77. "
        "По региональному коду определи регион РФ. "
        "Также предположи, в каком году он мог быть выдан."
        "Верни результат в JSON формате: { \"region_name\": \"...\"," 
        " \"year_issued\": \"...\" } без лишних слов."
    )
    user_msg = [
        {"type": "input_text", "text": f"Определи регион и год выдачи для номера: {plate_number}."}
    ]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    try:
        response = safe_chat_completion(messages, model="gpt-4o", temperature=0.1)
        if hasattr(response, 'output') and len(response.output) > 0:
            raw_resp = response.output[0].content[0].text.strip()
            # Пытаемся распарсить как JSON
            # Пример ожидаемого: { "region_name": "Московская область", "year_issued": "2015" }
            try:
                data = json.loads(raw_resp)
                region_name = data.get("region_name", "")
                year_issued = data.get("year_issued", "")
                return (region_name, year_issued)
            except:
                # Если не смогли распарсить
                return ("", "")
        else:
            return ("", "")
    except Exception as e:
        logging.error(f"[GPT-4o Error analyze_russian_number_gpt]: {e}")
        return ("", "")
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
# ИСПРАВЛЕНИЕ ФОРМАТА ДЛЯ НОВЫХ РФ НОМЕРОВ
###############################################################################
def fix_number_format_new_rus(plate_number: str, confs: list) -> (str, list):
    """
    Если видим лишнюю букву в конце - убираем
    + удаляем последнюю уверенность, чтобы длины совпадали.
    """
    valid_letters = "АВЕКМНОРСТУХ"
    pattern = r"^([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ]{2})(\d{2,3})$"

    if re.match(pattern, plate_number):
        return plate_number, confs

    # Проверяем последний символ
    if len(plate_number) > 0 and plate_number[-1] in valid_letters:
        candidate = plate_number[:-1]
        if re.match(pattern, candidate):
            # Убираем последний символ и последнюю уверенность
            plate_number = candidate
            if len(confs) > 0:
                confs = confs[:-1]
            return plate_number, confs

    return plate_number, confs

###############################################################################
# ИНТЕГРАЦИЯ: ГЕНЕРАЦИЯ JSON / XML
###############################################################################
def generate_json_result(plate_number, brand, car_type, color):
    data = {
        "plate_number": plate_number if plate_number else "",
        "brand": brand if brand else "",
        "car_type": car_type if car_type else "",
        "color": color if color else ""
    }
    return json.dumps(data, ensure_ascii=False, indent=4)

def generate_xml_result(plate_number, brand, car_type, color):
    root = ET.Element("CarAnalysisResult")

    plate_elem = ET.SubElement(root, "PlateNumber")
    plate_elem.text = plate_number if plate_number else ""

    brand_elem = ET.SubElement(root, "Brand")
    brand_elem.text = brand if brand else ""

    type_elem = ET.SubElement(root, "CarType")
    type_elem.text = car_type if car_type else ""

    color_elem = ET.SubElement(root, "Color")
    color_elem.text = color if color else ""

    xml_str = ET.tostring(root, encoding="utf-8")
    return xml_str.decode("utf-8")

###############################################################################
# ИМПОРТ JSON / XML
###############################################################################
def parse_json(json_str):
    """
    Возвращает кортеж (plate_number, brand, car_type, color).
    """
    try:
        data = json.loads(json_str)
        return (
            data.get("plate_number", ""),
            data.get("brand", ""),
            data.get("car_type", ""),
            data.get("color", "")
        )
    except:
        return ("", "", "", "")

def parse_xml(xml_str):
    """
    Возвращает кортеж (plate_number, brand, car_type, color).
    """
    try:
        root = ET.fromstring(xml_str)
        plate_number = root.findtext("PlateNumber", default="")
        brand = root.findtext("Brand", default="")
        car_type = root.findtext("CarType", default="")
        color = root.findtext("Color", default="")
        return (plate_number, brand, car_type, color)
    except:
        return ("", "", "", "")

###############################################################################
# ГЛОБАЛЬНАЯ СТАТИСТИКА
###############################################################################
# Будем хранить распознанные номера (plate_number), region, year_issued
# в st.session_state["stat_items"] как список словарей.
def init_stats():
    if "stat_items" not in st.session_state:
        st.session_state["stat_items"] = []

def add_stat_item(plate_number, region, year_issued):
    """
    Добавляем запись в нашу статистику.
    """
    st.session_state["stat_items"].append({
        "plate_number": plate_number,
        "region": region,
        "year_issued": year_issued
    })


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

###############################################################################
# ОСНОВНОЕ ПРИЛОЖЕНИЕ (STREAMLIT)
###############################################################################
def main():
    # Инициализируем статистику
    init_stats()

    st.set_page_config(page_title="AI Автоанализатор", layout="wide")
    st.title("Автомобильный Анализатор (AI)")

    # Сайдбар с настройками
    with st.sidebar:
        st.header("1) Загрузка и Настройки")

        uploaded_file = st.file_uploader("Выберите изображение автомобиля", type=["jpg", "jpeg", "png"])

        # Выбор формата номера
        number_format = st.selectbox(
            "Формат номера:",
            ['Старые РФ номера', 'Новые РФ номера', 'Зарубежные номера', 'Европейские номера', 'Австралийские номера', 'Американские номера'],
            index=1
        )

        # Выбор задачи
        task = st.radio(
            "Выберите задачу:",
            ['Распознать номер', 'Определить марку', 'Определить тип автомобиля', 'Определить цвет', 'Всё сразу']
        )

        # Порог уверенности для CNN
        confidence_threshold = st.slider(
            "Порог уверенности (CNN), ниже — символ = '?'",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

        # Кнопка Анализ
        analyze_button = st.button("Анализировать")

        st.write("---")
        st.header("2) Импорт результатов (JSON/XML)")
        imported_json_file = st.file_uploader("Импорт JSON", type=["json"], key="json_uploader")
        imported_xml_file = st.file_uploader("Импорт XML", type=["xml"], key="xml_uploader")

        import_button = st.button("Загрузить результаты из файла")

    # Основной экран: просмотр изображения
    if uploaded_file is not None:
        # Масштабируем изображение до фиксированной ширины (400 px), чтобы избежать
        # избыточного растягивания и не использовать "use_column_width"
        pil_img_raw = Image.open(uploaded_file)
        w_percent = (400 / float(pil_img_raw.width))
        h_size = int((float(pil_img_raw.height) * float(w_percent)))
        pil_img_resized = pil_img_raw.resize((400, h_size), Image.Resampling.LANCZOS)

        st.image(
            pil_img_resized,
            caption="Загруженное изображение (уменьшено)",
            # width=400 # можно и так, но уже вручную ресайзим => убираем
        )

    # Импорт JSON/XML
    if import_button:
        if imported_json_file is not None:
            json_str = imported_json_file.read().decode("utf-8")
            plate_i, brand_i, type_i, color_i = parse_json(json_str)
            st.success("Данные из JSON загружены:")
            st.write(f"Номер: {plate_i}, Марка: {brand_i}, Тип: {type_i}, Цвет: {color_i}")

        elif imported_xml_file is not None:
            xml_str = imported_xml_file.read().decode("utf-8")
            plate_i, brand_i, type_i, color_i = parse_xml(xml_str)
            st.success("Данные из XML загружены:")
            st.write(f"Номер: {plate_i}, Марка: {brand_i}, Тип: {type_i}, Цвет: {color_i}")
        else:
            st.warning("Не выбран файл для импорта.")

    # Запуск анализа
    if analyze_button and uploaded_file is not None:
        start_time = time.time()
        progress_bar = st.progress(0)

        # Считываем исходное изображение (в полном размере)
        pil_img = Image.open(uploaded_file)
        img_bgr = np.array(pil_img.convert("RGB"))[:, :, ::-1]

        plate_number, brand, car_type, color = None, None, None, None
        confs = []

        # РЕЖИМ "Всё сразу"
        if task == "Всё сразу":
            # Сначала распознаём номер
            plate_img, plate_part = plate_detect(img_bgr)
            st.subheader("Распознавание номера...")
            st.image(plate_img, caption="Обнаруженный номерной знак", width=300)

            plate_number, confs = recognize_number_cnn(plate_part, model_cnn, confidence_threshold)

            # Если "Новые РФ номера", исправляем
            if number_format == 'Новые РФ номера' and plate_number:
                plate_number, confs = fix_number_format_new_rus(plate_number, confs)

            progress_bar.progress(20)

            # Параллельно распознаём (марка, цвет, тип) с помощью GPT
            def task_brand():
                return recognize_brand_gpt4(img_bgr)

            def task_color():
                return recognize_color_gpt4(img_bgr)

            def task_type():
                return classify_car_type(img_bgr)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_brand = executor.submit(task_brand)
                future_color = executor.submit(task_color)
                future_type = executor.submit(task_type)

                brand = future_brand.result()
                color = future_color.result()
                car_type = future_type.result()

            progress_bar.progress(90)

        else:
            # РЕЖИМЫ по отдельности
            # 1) Номер
            if task == 'Распознать номер':
                st.subheader("Распознавание номера...")
                plate_img, plate_part = plate_detect(img_bgr)
                st.image(plate_img, caption="Обнаруженный номерной знак", width=300)
                plate_number, confs = recognize_number_cnn(plate_part, model_cnn, confidence_threshold)

                # "Новые РФ" -> исправляем
                if number_format == 'Новые РФ номера' and plate_number:
                    plate_number, confs = fix_number_format_new_rus(plate_number, confs)
                progress_bar.progress(50)

            # 2) Марка
            if task == 'Определить марку':
                st.subheader("Определение марки...")
                brand = recognize_brand_gpt4(img_bgr)
                progress_bar.progress(50)

            # 3) Тип
            if task == 'Определить тип автомобиля':
                st.subheader("Определение типа...")
                car_type = classify_car_type(img_bgr)
                progress_bar.progress(50)

            # 4) Цвет
            if task == 'Определить цвет':
                st.subheader("Определение цвета...")
                color = recognize_color_gpt4(img_bgr)
                progress_bar.progress(50)

        # Если это номер РФ (старый / новый), проведём дополнительный анализ (регион, год)
        region_name = ""
        year_issued = ""
        if plate_number and (number_format in ['Старые РФ номера', 'Новые РФ номера']):
            # Запрашиваем GPT
            st.subheader("Дополнительный анализ номера (регион, год выдачи)")
            region_name, year_issued = analyze_russian_number_gpt(plate_number)

        progress_bar.progress(100)

        elapsed_time = time.time() - start_time
        logging.info(f"Обработка завершена за {elapsed_time:.2f} сек.")
        st.success(f"Обработка заняла {elapsed_time:.2f} сек.")

        # Итоговые результаты
        st.header("Результаты анализа")

        if plate_number:
            st.write(f"**Распознанный номер**: {plate_number}")
            if region_name:
                st.write(f"**Регион**: {region_name}")
            if year_issued:
                st.write(f"**Год выдачи**: {year_issued}")

            # Добавляем запись в статистику
            add_stat_item(plate_number, region_name, year_issued)

            # Покажем уверенность распознавания
            if len(confs) > 0:
                avg_conf = sum(confs) / len(confs)
                st.write(f"Средняя уверенность (CNN): {avg_conf:.2f}")
                if len(plate_number) == len(confs):
                    df_conf = pd.DataFrame({
                        "Символ": list(plate_number),
                        "Уверенность": confs
                    })
                    st.bar_chart(df_conf, x="Символ", y="Уверенность", height=200)
                else:
                    st.warning("Невозможно отобразить детализацию по символам. Проверьте сегментацию.")

        if brand:
            st.write(f"**Марка**: {brand}")

        if car_type:
            st.write(f"**Тип**: {car_type}")

        if color:
            st.write(f"**Цвет**: {color}")

        # Экспорт результатов
        st.subheader("Выгрузить результат в JSON/XML")
        result_json = generate_json_result(plate_number, brand, car_type, color)
        st.download_button(
            label="Скачать JSON",
            data=result_json.encode('utf-8'),
            file_name="result.json",
            mime="application/json"
        )
        result_xml = generate_xml_result(plate_number, brand, car_type, color)
        st.download_button(
            label="Скачать XML",
            data=result_xml.encode('utf-8'),
            file_name="result.xml",
            mime="application/xml"
        )

    elif analyze_button and not uploaded_file:
        st.warning("Сначала загрузите изображение.")

    # Отображаем таблицу статистики, если есть
    st.write("---")
    st.header("Статистика распознанных номеров (РФ)")
    if len(st.session_state["stat_items"]) > 0:
        df_stats = pd.DataFrame(st.session_state["stat_items"])
        st.dataframe(df_stats)
    else:
        st.write("Пока нет записей в статистике.")

###############################################################################
# Запуск
###############################################################################
if __name__ == '__main__':
    # Если runtime существует, запускаем напрямую
    if runtime.exists():
        main()
    else:
        # Иначе запускаем через streamlit run
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
