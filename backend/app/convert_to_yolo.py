# convert_to_yolo.py - Конвертация разметок в формат YOLO
import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np

print("=" * 70)
print("🤖 КОНВЕРТАЦИЯ РАЗМЕТОК В ФОРМАТ YOLO ДЛЯ ОБУЧЕНИЯ")
print("=" * 70)

# ПУТИ (ИСПРАВЛЕНО!)
IMAGES_DIR = Path("C:/smet4ik/backend/processed_images")  # ЗДЕСЬ ВАШИ ИЗОБРАЖЕНИЯ!
MARKUPS_DIR = Path("C:/smet4ik/backend/app/markups")      # ЗДЕСЬ ВАШИ РАЗМЕТКИ!
YOLO_DIR = Path("C:/smet4ik/backend/yolo_dataset")        # СЮДА СОХРАНИМ ДАННЫЕ ДЛЯ YOLO

# Создаем папки для YOLO датасета
YOLO_DIR.mkdir(exist_ok=True)
(YOLO_DIR / "images").mkdir(exist_ok=True)
(YOLO_DIR / "labels").mkdir(exist_ok=True)
(YOLO_DIR / "images/train").mkdir(exist_ok=True)
(YOLO_DIR / "images/val").mkdir(exist_ok=True)
(YOLO_DIR / "labels/train").mkdir(exist_ok=True)
(YOLO_DIR / "labels/val").mkdir(exist_ok=True)

# Классы для YOLO (пока только стена)
CLASSES = ["wall"]  # 0: wall
class_to_id = {"wall": 0}

print(f"📁 Изображения: {IMAGES_DIR}")
print(f"📁 Разметки: {MARKUPS_DIR}")
print(f"📁 YOLO датасет: {YOLO_DIR}")
print(f"🎯 Классы: {CLASSES}")

# Собираем все разметки
markup_files = list(MARKUPS_DIR.rglob("*.json"))
print(f"\n📊 Найдено файлов разметок: {len(markup_files)}")

if not markup_files:
    print("❌ Нет файлов разметок!")
    exit()

# Функция для поиска изображения по project_id и page_num
def find_image(project_id, page_num):
    """Ищет изображение для проекта и страницы"""
    
    # Пробуем разные варианты имен файлов
    possible_patterns = [
        f"page_{page_num:03d}.jpg",
        f"page_{page_num}.jpg",
        f"page_{page_num:03d}.png",
        f"page_{page_num}.png",
        f"{page_num}.jpg",
        f"{page_num}.png"
    ]
    
    project_image_dir = IMAGES_DIR / project_id
    
    if not project_image_dir.exists():
        print(f"    ❌ Папка проекта не найдена: {project_image_dir}")
        return None
    
    # Ищем по шаблонам
    for pattern in possible_patterns:
        image_path = project_image_dir / pattern
        if image_path.exists():
            return image_path
    
    # Если не нашли по шаблону, берем первый файл изображения
    all_images = list(project_image_dir.glob("*.jpg")) + list(project_image_dir.glob("*.png"))
    if all_images:
        # Сортируем и берем по номеру страницы
        all_images.sort()
        if page_num <= len(all_images):
            return all_images[page_num - 1]
    
    return None

# Функция конвертации координат в YOLO формат
def convert_to_yolo_format(points, image_width, image_height):
    """Конвертирует полигон в YOLO формат (нормализованные координаты)"""
    
    if not points or len(points) < 2:
        return []
    
    # Находим bounding box (ограничивающий прямоугольник)
    x_coords = [p["x"] for p in points]
    y_coords = [p["y"] for p in points]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Центр bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Ширина и высота
    width = x_max - x_min
    height = y_max - y_min
    
    # Нормализуем (делим на размеры изображения)
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

# Обрабатываем каждую разметку
success_count = 0
error_count = 0

for markup_file in markup_files:
    print(f"\n🔍 Обрабатываю: {markup_file.name}")
    
    try:
        # Читаем разметку
        with open(markup_file, "r", encoding="utf-8") as f:
            markup = json.load(f)
        
        # Получаем информацию о проекте
        project_id = markup.get("project_id", "")
        page_num = markup.get("page_num", 1)
        
        if not project_id:
            # Пробуем извлечь из имени файла
            project_id = markup_file.parent.name
        
        print(f"   Проект: {project_id}, Страница: {page_num}")
        
        # Ищем изображение
        image_path = find_image(project_id, page_num)
        
        if not image_path:
            print(f"   ❌ Изображение не найдено для проекта {project_id}")
            error_count += 1
            continue
        
        print(f"   ✅ Изображение: {image_path.name}")
        
        # Читаем изображение чтобы получить размеры
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"   ❌ Не удалось прочитать изображение")
            error_count += 1
            continue
        
        height, width = img.shape[:2]
        print(f"   Размеры: {width}x{height} пикселей")
        
        # Подготавливаем данные для YOLO
        yolo_lines = []
        
        if "objects" in markup:
            for obj in markup["objects"]:
                obj_type = obj.get("type", "")
                
                # Пока обрабатываем только стены
                if obj_type == "wall" and "points" in obj:
                    points = obj["points"]
                    
                    # Конвертируем в YOLO формат
                    yolo_coords = convert_to_yolo_format(points, width, height)
                    
                    if yolo_coords:
                        class_id = class_to_id.get(obj_type, 0)
                        line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in yolo_coords)
                        yolo_lines.append(line)
                        print(f"   🧱 Добавлена стена: {len(points)} точек")
        
        if not yolo_lines:
            print(f"   ⚠️ Нет стен в разметке")
            error_count += 1
            continue
        
        # Копируем изображение в YOLO датасет
        image_name = f"{project_id}_p{page_num}.jpg"
        dest_image_path = YOLO_DIR / "images/train" / image_name
        
        # Конвертируем в JPG если нужно
        if image_path.suffix.lower() in ['.png', '.jpeg']:
            # Сохраняем как JPG
            cv2.imwrite(str(dest_image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"   💾 Конвертировано в JPG: {image_name}")
        else:
            # Копируем как есть
            shutil.copy2(image_path, dest_image_path)
            print(f"   💾 Скопировано изображение: {image_name}")
        
        # Сохраняем разметку в YOLO формате
        label_name = f"{project_id}_p{page_num}.txt"
        dest_label_path = YOLO_DIR / "labels/train" / label_name
        
        with open(dest_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
        
        print(f"   📝 Сохранено аннотаций: {len(yolo_lines)}")
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Ошибка обработки {markup_file}: {e}")
        error_count += 1

print("\n" + "=" * 70)
print("📊 РЕЗУЛЬТАТ КОНВЕРТАЦИИ:")
print("=" * 70)
print(f"✅ Успешно обработано: {success_count} разметок")
print(f"❌ Ошибок: {error_count}")

if success_count > 0:
    print(f"\n🎉 ДАННЫЕ ГОТОВЫ ДЛЯ ОБУЧЕНИЯ YOLO!")
    print(f"📁 Папка с датасетом: {YOLO_DIR}")
    
    # Создаем файл dataset.yaml
    yaml_content = f"""# YOLO Dataset Configuration
path: {YOLO_DIR}  # dataset root dir
train: images/train  # train images
val: images/val  # val images

# Classes
names:
  0: wall
"""

    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"📄 Создан файл конфигурации: {yaml_path}")
    
    print(f"\n📋 Содержимое датасета:")
    print(f"   📸 Изображений для обучения: {len(list((YOLO_DIR / 'images/train').glob('*.jpg')))}")
    print(f"   📝 Аннотаций для обучения: {len(list((YOLO_DIR / 'labels/train').glob('*.txt')))}")
    
    print(f"\n🚀 ВЫ МОЖЕТЕ НАЧАТЬ ОБУЧЕНИЕ YOLO!")
    
else:
    print("\n❌ НЕ УДАЛОСЬ ПОДГОТОВИТЬ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ")

print("\nНажмите Enter для продолжения...")
input()
