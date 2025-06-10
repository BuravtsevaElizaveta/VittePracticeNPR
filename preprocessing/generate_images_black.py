from PIL import Image, ImageDraw, ImageFont
import os

# Путь для сохранения шаблонов
template_dir = 'templates'

# Создание директории, если она не существует
if not os.path.exists(template_dir):
    os.makedirs(template_dir)

# Список символов для российских номерных знаков
characters = "АВЕКМНОРСТУХ0123456789"

# Параметры для создания изображений символов
font_size = 48
img_size = (50, 50)  # Размер изображения символа

# Путь к шрифту, поддерживающему кириллицу
font_path = "arial.ttf"

# Функция для создания шаблона символа
def create_template(char):
    img = Image.new('1', img_size, color=1)  # Белый фон
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), char, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_size[0] - text_width) // 2
    text_y = (img_size[1] - text_height) // 2
    draw.text((text_x, text_y), char, font=font, fill=0)  # Черный текст
    return img

# Создание и сохранение шаблонов
for char in characters:
    template = create_template(char)
    template.save(os.path.join(template_dir, f"{char}.png"))

print("Шаблоны созданы и сохранены в:", template_dir)
