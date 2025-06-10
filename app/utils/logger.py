# utils/logger.py

import logging
import os

def setup_logger():
    """
    Настраивает логгер для приложения.

    Возвращает:
    - logger: Настроенный экземпляр логгера.
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger('yolo_app')
    logger.setLevel(logging.DEBUG)
    
    # Создание файлового обработчика, который записывает все уровни логов
    fh = logging.FileHandler('logs/app.log')
    fh.setLevel(logging.DEBUG)
    
    # Создание консольного обработчика с уровнем ERROR
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    
    # Создание форматтера и добавление его к обработчикам
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Добавление обработчиков к логгеру
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger()
