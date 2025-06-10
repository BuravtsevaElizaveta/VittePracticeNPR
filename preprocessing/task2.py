# Импортируем библиотеку OpenCV для обработки изображений
import cv2
# Импортируем библиотеку NumPy для работы с массивами
import numpy as np
# Импортируем библиотеку os для работы с файловой системой
import os
# Импортируем библиотеку pytesseract для оптического распознавания символов (OCR)
import pytesseract
# Импортируем библиотеку io для работы с потоками ввода-вывода
import io
# Импортируем библиотеку time для замера времени выполнения операций
import time

# Указываем путь к исполняемому файлу Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
# Конфигурация для Tesseract OCR, указывающая путь к данным
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'

# Функция для нахождения номерной пластины на изображении
def find_plate(image_path):
    # Загружаем каскад классификатора для распознавания номерных пластин
    cascade_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    # Открываем изображение в бинарном режиме и загружаем его в поток
    with open(image_path, 'rb') as f:
        img_stream = io.BytesIO(f.read())
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Проверяем, удалось ли загрузить изображение
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ищем номерные пластины на изображении
    plates = cascade_plate.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Возвращаем изображение с номерной пластиной, если она найдена
    for (x, y, w, h) in plates:
        plate_img = img[y:y+h, x:x+w]
        return plate_img
    return None

# Функция для предобработки изображения
def preprocess_image(img):
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Применяем бинаризацию с инверсией и методом Отсу
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# Функция для увеличения изображения
def enlarge_img(image, scale_percent):
    # Рассчитываем новые размеры изображения
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Изменяем размер изображения с интерполяцией по площади
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Функция для распознавания текста с использованием Tesseract OCR
def recognize_text_with_tesseract(img):
    # Конфигурация для Tesseract OCR
    config = '--psm 8 -c tessedit_char_whitelist=ABEKMHOPCTYXАВЕКМНОРСТУХ0123456789 -l rus'
    # Распознаем текст на изображении
    text = pytesseract.image_to_string(img, config=config + " " + tessdata_dir_config)
    # Фильтруем распознанный текст, оставляя только разрешенные символы
    return ''.join(filter(lambda x: x in "ABEKMHOPCTYXАВЕКМНОРСТУХ0123456789", text.strip()))

# Функция для замены эквивалентных символов
def replace_equivalent_chars(text):
    # Словарь замен эквивалентных символов
    replacements = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н', 'O': 'О',
        'P': 'Р', 'C': 'С', 'T': 'Т', 'Y': 'У', 'X': 'Х',
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
        'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X'
    }
    # Возвращаем текст с замененными символами
    return ''.join(replacements.get(char, char) for char in text)

# Функция для проверки точности распознавания
def check_accuracy(recognized_text, expected_text):
    # Заменяем эквивалентные символы в распознанном и ожидаемом тексте
    recognized_text = replace_equivalent_chars(recognized_text)
    expected_text = replace_equivalent_chars(expected_text)
    
    # Считаем количество совпадающих символов
    correct_chars = sum(1 for a, b in zip(recognized_text, expected_text) if a == b)
    length = len(recognized_text)
    
    # Рассчитываем точность распознавания
    accuracy = correct_chars / length * 100 if length > 0 else 0
    # Проверяем, полностью ли совпадают тексты
    full_match = recognized_text == expected_text
    
    return accuracy, full_match

# Обработка всех изображений в папке test
test_dir = 'test'
# Получаем список изображений в папке test
test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png'))]

# Инициализируем переменные для подсчета общей точности и количества совпадений
total_correct_chars = 0
total_chars = 0
total_full_matches = 0
num_images = len(test_images)

# Обрабатываем каждое изображение в папке test
for test_image in test_images:
    try:
        # Формируем полный путь к изображению
        image_path = os.path.join(test_dir, test_image)
        # Ожидаемый текст - имя файла без расширения, переведенное в верхний регистр
        expected_text = os.path.splitext(test_image)[0].upper()
        
        # Измеряем время нахождения номерной пластины
        start_time = time.time()
        plate_img = find_plate(image_path)
        end_time = time.time()
        plate_detection_time = end_time - start_time
        
        if plate_img is not None:
            # Измеряем время предобработки изображения
            start_time = time.time()
            enlarged_img = enlarge_img(plate_img, 150)
            gray_img = cv2.cvtColor(enlarged_img, cv2.COLOR_BGR2GRAY)
            blurred_img = cv2.medianBlur(gray_img, 3)
            _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            end_time = time.time()
            preprocessing_time = end_time - start_time

            # Измеряем время распознавания OCR
            start_time = time.time()
            recognized_text = recognize_text_with_tesseract(binary_img)
            end_time = time.time()
            ocr_time = end_time - start_time

            # Заменяем эквивалентные символы в распознанном и ожидаемом тексте
            recognized_text = replace_equivalent_chars(recognized_text)
            expected_text = replace_equivalent_chars(expected_text)
            
            # Проверяем точность распознавания
            accuracy, full_match = check_accuracy(recognized_text, expected_text)
            
            # Подсчитываем количество совпадающих символов и обновляем общие показатели
            correct_chars = sum(1 for a, b in zip(recognized_text, expected_text) if a == b)
            total_correct_chars += correct_chars
            total_chars += len(recognized_text)
            total_full_matches += full_match
            
            # Выводим результаты для текущего изображения
            print(f"Файл: {test_image}")
            print(f"Распознанный текст: {recognized_text}")
            print(f"Ожидаемый текст: {expected_text}")
            print(f"Точность распознавания: {accuracy:.2f}%")
            print(f"Полное совпадение: {'Да' if full_match else 'Нет'}")
            print(f"Время нахождения пластины: {plate_detection_time:.4f} секунд")
            print(f"Время предобработки: {preprocessing_time:.4f} секунд")
            print(f"Время распознавания OCR: {ocr_time:.4f} секунд")
            print()
        else:
            # Выводим сообщение, если номерная пластина не найдена
            print(f"Номерная пластина не найдена в изображении {test_image}")
    except Exception as e:
        # Обрабатываем ошибки и выводим сообщение об ошибке
        print(f"Ошибка при обработке изображения {test_image}: {e}")

# Рассчитываем и выводим общую точность распознавания
if total_chars > 0:
    overall_accuracy = total_correct_chars / total_chars * 100
else:
    overall_accuracy = 0

print(f"Общая точность распознавания: {overall_accuracy:.2f}%")
print(f"Количество полных совпадений: {total_full_matches} из {num_images}")
