# utils/database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Detection(Base):
    """
    Класс для таблицы детекций в базе данных.
    """
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    object_class = Column(String)
    quantity = Column(Integer)

def init_db(db_path='sqlite:///detections.db'):
    """
    Инициализирует базу данных.

    Параметры:
    - db_path (str): Строка подключения к базе данных.

    Возвращает:
    - session: Сессия SQLAlchemy.
    """
    try:
        logger.info(f"Инициализация базы данных по адресу: {db_path}")
        engine = create_engine(db_path)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        logger.info("База данных успешно инициализирована.")
        return Session()
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных: {e}")
        raise e

def log_detection(object_class, quantity, db_session):
    """
    Логирует одну детекцию в базу данных.

    Параметры:
    - object_class (str): Класс обнаруженного объекта.
    - quantity (int): Количество обнаруженных объектов.
    - db_session: Сессия базы данных.
    """
    try:
        detection = Detection(object_class=object_class, quantity=quantity)
        db_session.add(detection)
        db_session.commit()
        logger.info(f"Логирование детекции: {object_class} - {quantity}")
    except Exception as e:
        logger.error(f"Ошибка при логировании детекции: {e}")
        db_session.rollback()

def log_detections_bulk(detections, db_session):
    """
    Логирует несколько детекций в базу данных.

    Параметры:
    - detections (dict): Словарь с классами объектов и их количеством.
    - db_session: Сессия базы данных.
    """
    try:
        logger.info("Начало логирования нескольких детекций.")
        for object_class, quantity in detections.items():
            detection = Detection(object_class=object_class, quantity=quantity)
            db_session.add(detection)
        db_session.commit()
        logger.info(f"Логирование нескольких детекций: {detections}")
    except Exception as e:
        logger.error(f"Ошибка при логировании нескольких детекций: {e}")
        db_session.rollback()
