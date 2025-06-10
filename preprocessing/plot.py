import matplotlib.pyplot as plt
import pandas as pd

# Прочитаем лог-файлы
log_task1 = 'log_task1.txt'
log_task2 = 'log_task2.txt'

# Функция для парсинга лог-файлов
def parse_log(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if "Файл:" in line:
                file_name = line.split()[-1]
            if "Распознанный текст:" in line:
                recognized_text = line.split(": ")[1].strip()
            if "Ожидаемый текст:" in line:
                expected_text = line.split(": ")[1].strip()
            if "Точность распознавания:" in line:
                accuracy = float(line.split(": ")[1].strip().strip('%'))
            if "Полное совпадение:" in line:
                full_match = line.split(": ")[1].strip() == "Да"
            if "Время нахождения пластины:" in line:
                plate_detection_time = float(line.split(": ")[1].strip().split()[0])
            if "Время предобработки:" in line:
                preprocessing_time = float(line.split(": ")[1].strip().split()[0])
            if "Время распознавания OCR:" in line:
                ocr_time = float(line.split(": ")[1].strip().split()[0])
                # Добавляем результат в список
                results.append((file_name, recognized_text, expected_text, accuracy, full_match, plate_detection_time, preprocessing_time, ocr_time))
    return results

# Парсинг логов
results_task1 = parse_log(log_task1)
results_task2 = parse_log(log_task2)

# Преобразуем результаты в DataFrame
columns = ["Файл", "Распознанный текст", "Ожидаемый текст", "Точность", "Полное совпадение", "Время нахождения пластины", "Время предобработки", "Время распознавания OCR"]
df_task1 = pd.DataFrame(results_task1, columns=columns)
df_task2 = pd.DataFrame(results_task2, columns=columns)

# Построение графиков

# График точности распознавания для каждого изображения
plt.figure(figsize=(15, 5))
plt.plot(df_task1['Файл'], df_task1['Точность'], label='сопоставление с шаблоном')
plt.plot(df_task2['Файл'], df_task2['Точность'], label='компьютерное зрение')
plt.xlabel('Изображение')
plt.ylabel('Точность (%)')
plt.title('Точность распознавания для каждого изображения')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# График времени выполнения различных этапов обработки
plt.figure(figsize=(15, 5))
plt.plot(df_task1['Файл'], df_task1['Время нахождения пластины'], label='Время нахождения пластины - сопоставление с шаблоном')
plt.plot(df_task1['Файл'], df_task1['Время предобработки'], label='Время предобработки - сопоставление с шаблоном')
plt.plot(df_task1['Файл'], df_task1['Время распознавания OCR'], label='Время распознавания OCR - сопоставление с шаблоном')
plt.plot(df_task2['Файл'], df_task2['Время нахождения пластины'], label='Время нахождения пластины - компьютерное зрение')
plt.plot(df_task2['Файл'], df_task2['Время предобработки'], label='Время предобработки - компьютерное зрение')
plt.plot(df_task2['Файл'], df_task2['Время распознавания OCR'], label='Время распознавания OCR - компьютерное зрение')
plt.xlabel('Изображение')
plt.ylabel('Время (секунды)')
plt.title('Время выполнения различных этапов обработки')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# График общего сравнения двух алгоритмов
data = {
    'Метрика': ['Общая точность', 'Полные совпадения'],
    'сопоставление с шаблоном': [df_task1['Точность'].mean(), df_task1['Полное совпадение'].sum()],
    'компьютерное зрение': [df_task2['Точность'].mean(), df_task2['Полное совпадение'].sum()]
}
df_comparison = pd.DataFrame(data)

df_comparison.plot(x='Метрика', kind='bar', figsize=(10, 5))
plt.title('Общее сравнение двух алгоритмов')
plt.ylabel('Значение')
plt.show()
