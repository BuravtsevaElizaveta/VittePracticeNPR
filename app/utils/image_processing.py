# utils/image_processing.py

import cv2
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

def draw_boxes(image, detections, conf_thresh, model, line_thickness):
    """
    Рисует bounding boxes и метки на изображении.

    Параметры:
    - image: Изображение/кадр для рисования (numpy array).
    - detections: Список детекций (numpy array).
    - conf_thresh (float): Порог уверенности.
    - model: Загруженная модель YOLO.
    - line_thickness (int): Толщина линий для bounding boxes.

    Возвращает:
    - image: Изображение с нарисованными bounding boxes.
    """
    try:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = model.model.names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            color = (0, 255, 0)  # Зеленый цвет для bounding boxes
            # Рисование прямоугольника
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
            # Рисование фона для метки
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
            # Рисование текста метки
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return image
    except Exception as e:
        logger.error(f"Ошибка при рисовании bounding boxes: {e}")
        return image

def process_image(image, model, conf_thresh, line_thickness):
    """
    Обрабатывает одно изображение для детекции объектов.

    Параметры:
    - image: Загруженное изображение (PIL Image).
    - model: Загруженная модель YOLO.
    - conf_thresh (float): Порог уверенности.
    - line_thickness (int): Толщина линий для bounding boxes.

    Возвращает:
    - output_image: Изображение с детекциями.
    - detections: Список детекций.
    """
    try:
        logger.info("Начало обработки загруженного изображения.")
        # Конвертация изображения в формат BGR для OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Выполнение детекции
        detections = model.predict(img, conf_thresh)
        logger.debug(f"Детекции: {detections}")
        # Рисование bounding boxes
        output_image = draw_boxes(img, detections, conf_thresh, model, line_thickness)
        # Конвертация обратно в RGB для отображения в Streamlit
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)
        logger.info("Обработка изображения завершена.")
        return output_image, detections
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        raise e
