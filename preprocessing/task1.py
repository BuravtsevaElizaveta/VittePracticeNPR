# Импорт необходимых библиотек
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
import io
import time

# Функция для загрузки шаблонов символов
def load_templates(template_dir):
    templates = {}
    # Получение списка файлов в директории с шаблонами
    files = os.listdir(template_dir)
    for filename in files:
        # Проверка на соответствие имени файла допустимым форматам изображений
        if filename.lower().endswith(('.jpg', '.png')):
            # Получение имени символа из имени файла
            char = os.path.splitext(filename)[0]
            # Полный путь к файлу
            file_path = os.path.join(template_dir, filename)
            try:
                # Загрузка изображения шаблона
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[char] = img
                    print(f"Loaded template for character {char}")
                else:
                    print(f"Error loading template for character {char}")
            except Exception as e:
                print(f"Exception loading template for character {char}: {e}")
    return templates

# Функция для нахождения номерной пластины на изображении
def find_plate(image_path):
    # Загрузка классификатора для распознавания номерных пластин
    cascade_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    with open(image_path, 'rb') as f:
        img_stream = io.BytesIO(f.read())
        file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        # Декодирование изображения из байтового потока
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Поиск номерных пластин на изображении
    plates = cascade_plate.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in plates:
        # Обрезка изображения до области с номерной пластиной
        plate_img = img[y:y+h, x:x+w]
        return plate_img
    return None

# Функция для предобработки изображения
def preprocess_image(img):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Задание размера блока для адаптивного порогового значения
    block_size = 35
    # Применение локального порогового значения
    adaptive_thresh = threshold_local(gray, block_size, offset=10)
    # Бинаризация изображения
    binary = (gray > adaptive_thresh).astype(np.uint8) * 255
    return binary

# Функция для сегментации символов на изображении
def segment_characters(binary_img):
    # Поиск контуров на бинаризованном изображении
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for cnt in contours:
        # Получение координат ограничивающего прямоугольника для каждого контура
        x, y, w, h = cv2.boundingRect(cnt)
        segments.append((x, y, w, h))
    # Сортировка сегментов по горизонтальной оси (слева направо)
    segments = sorted(segments, key=lambda x: x[0])
    return segments

# Функция для сопоставления сегментов с шаблонами с использованием cv2.matchTemplate
def match_templates(binary_img, segments, templates):
    recognized_text = ""
    for (x, y, w, h) in segments:
        # Извлечение области интереса (ROI) для текущего сегмента
        roi = binary_img[y:y+h, x:x+w]
        best_match = None
        best_score = -1  # Начальное значение для лучшей корреляции
        for char, template in templates.items():
            if template is not None:
                try:
                    # Изменение размера ROI до размеров шаблона
                    resized_roi = cv2.resize(roi, (template.shape[1], template.shape[0]))
                    # Сопоставление ROI с шаблоном
                    result = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > best_score:
                        best_match = char
                        best_score = max_val
                except Exception as e:
                    print(f"Error in match_templates for character {char}: {e}")
                    continue
        print(f"Segment {(x, y, w, h)}: Best match = {best_match}, Score = {best_score}")
        recognized_text += best_match if best_match else ''
    return recognized_text

# Функция для удаления групп из 3 и более одинаковых символов
def remove_repeated_groups(text):
    result = []
    count = 1
    prev_char = ''
    for char in text:
        if char == prev_char:
            count += 1
        else:
            if count < 3:
                result.extend([prev_char] * count)
            count = 1
        prev_char = char
    if count < 3:
        result.extend([prev_char] * count)
    return ''.join(result)

# Функция для фильтрации допустимых символов и ограничения длины текста
def filter_valid_characters(text):
    valid_chars = "ABEIKMHOPCTYXАВЕКМНОРСТУХ0123456789"
    filtered_text = ''.join([char for char in text if char in valid_chars])
    return filtered_text[:10]

# Функция для замены эквивалентных символов
def replace_equivalent_chars(text):
    replacements = {
        'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М', 'H': 'Н', 'O': 'О',
        'P': 'Р', 'C': 'С', 'T': 'Т', 'Y': 'У', 'X': 'Х',
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
        'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X'
    }
    return ''.join(replacements.get(char, char) for char in text)

# Функция для проверки точности распознавания
def check_accuracy(recognized_text, expected_text):
    # Замена эквивалентных символов в распознанном и ожидаемом тексте
    recognized_text = replace_equivalent_chars(recognized_text)
    expected_text = replace_equivalent_chars(expected_text)
    
    # Подсчет количества правильно распознанных символов
    correct_chars = sum(1 for a, b in zip(recognized_text, expected_text) if a == b)
    length = len(recognized_text)
    
    # Вычисление точности распознавания в процентах
    accuracy = correct_chars / length * 100 if length > 0 else 0
    # Проверка на полное совпадение текста
    full_match = recognized_text == expected_text
    
    return accuracy, full_match

# Функция для визуализации сегментов на изображении
def visualize_segments(binary_img, segments):
    # Преобразование бинарного изображения в цветное для отображения
    img_copy = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in segments:
        # Рисование прямоугольников вокруг каждого сегмента
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()

# Загрузка шаблонов символов
template_dir = 'templates_inverted'
templates = load_templates(template_dir)

# Обработка всех изображений в папке test
test_dir = 'test'
# Получение списка всех тестовых изображений
test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png'))]

total_correct_chars = 0
total_chars = 0
total_full_matches = 0
num_images = len(test_images)

# Обработка каждого тестового изображения
for test_image in test_images:
    try:
        image_path = os.path.join(test_dir, test_image)
        # Ожидаемый текст на основе имени файла (без расширения)
        expected_text = os.path.splitext(test_image)[0].upper()
        
        start_time = time.time()
        # Нахождение номерной пластины на изображении
        plate_img = find_plate(image_path)
        end_time = time.time()
        plate_detection_time = end_time - start_time
        
        if plate_img is not None:
            start_time = time.time()
            # Предобработка изображения
            binary_img = preprocess_image(plate_img)
            end_time = time.time()
            preprocessing_time = end_time - start_time

            # Сегментация символов
            segments = segment_characters(binary_img)
            # Распознавание символов на основе шаблонов
            recognized_text = match_templates(binary_img, segments, templates)

            start_time = time.time()
            # Постобработка распознанного текста
            recognized_text = remove_repeated_groups(recognized_text)
            recognized_text = filter_valid_characters(recognized_text)
            end_time = time.time()
            post_processing_time = end_time - start_time

            # Проверка точности распознавания
            accuracy, full_match = check_accuracy(recognized_text, expected_text)
            
            # Подсчет количества правильно распознанных символов
            correct_chars = sum(1 for a, b in zip(recognized_text, expected_text) if a == b)
            total_correct_chars += correct_chars
            total_chars += len(recognized_text)
            total_full_matches += full_match
            
            # Вывод результатов распознавания для текущего изображения
            print(f"Файл: {test_image}")
            print(f"Распознанный текст: {recognized_text}")
            print(f"Ожидаемый текст: {expected_text}")
            print(f"Точность распознавания: {accuracy:.2f}%")
            print(f"Полное совпадение: {'Да' if full_match else 'Нет'}")
            print(f"Время нахождения пластины: {plate_detection_time:.4f} секунд")
            print(f"Время предобработки: {preprocessing_time:.4f} секунд")
            print(f"Время постобработки: {post_processing_time:.4f} секунд")
            print()
        else:
            print(f"Номерная пластина не найдена в изображении {test_image}")
    except Exception as e:
        print(f"Ошибка при обработке изображения {test_image}: {e}")

# Вычисление общей точности распознавания для всех изображений
if total_chars > 0:
    overall_accuracy = total_correct_chars / total_chars * 100
else:
    overall_accuracy = 0

# Вывод общей статистики
print(f"Общая точность распознавания: {overall_accuracy:.2f}%")
print(f"Количество полных совпадений: {total_full_matches} из {num_images}")
