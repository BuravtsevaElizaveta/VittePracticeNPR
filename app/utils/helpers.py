# utils/helpers.py

import logging

logger = logging.getLogger(__name__)

def format_detections(detections, model):
    """
    Форматирует детекции для логирования и отображения.

    Параметры:
    - detections: Список детекций от модели (numpy array).
    - model: Загруженная модель YOLO.

    Возвращает:
    - count_dict (dict): Словарь с классами объектов и их количеством.
    """
    try:
        count_dict = {}
        for det in detections:
            cls_id = int(det[5])
            class_name = model.model.names[cls_id]
            count_dict[class_name] = count_dict.get(class_name, 0) + 1
        logger.debug(f"Форматированные детекции: {count_dict}")
        return count_dict
    except Exception as e:
        logger.error(f"Ошибка при форматировании детекций: {e}")
        return {}
