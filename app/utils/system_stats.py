# utils/system_stats.py

import psutil
import logging

logger = logging.getLogger(__name__)

def get_system_stats():
    """
    Получает текущую системную статистику по памяти и CPU.

    Возвращает:
    - stats (dict): Словарь с процентом использования памяти и CPU.
    """
    try:
        logger.debug("Получение системной статистики.")
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        stats = {
            "memory_percent": memory.percent,
            "cpu_percent": cpu
        }
        logger.debug(f"Системная статистика: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Ошибка при получении системной статистики: {e}")
        return {"memory_percent": "N/A", "cpu_percent": "N/A"}

def get_disk_usage(path):
    """
    Получает использование диска для указанного пути.

    Параметры:
    - path (str): Путь для проверки использования диска.

    Возвращает:
    - usage (dict): Словарь с процентом использования диска.
    """
    try:
        usage = psutil.disk_usage(path)._asdict()
        logger.debug(f"Использование диска для {path}: {usage}")
        return usage
    except Exception as e:
        logger.error(f"Ошибка при получении использования диска: {e}")
        return {"percent": "N/A"}
