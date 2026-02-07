# convert_to_yolo_fixed.py - Исправленная конвертация с объединением разметок
import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

print("=" * 70)
print("🤖 ИСПРАВЛЕННАЯ КОНВЕРТАЦИЯ - ОБЪЕДИНЕНИЕ ВСЕХ РАЗМЕТОК")
print("=" * 70)

# ПУТИ
IMAGES_DIR = Path("C:/smet4ik/backend/processed_images")
MARKUPS_DIR = Path("C:/smet4ik/backend/app/markups")
YOLO_DIR = Path("C:/smet4ik/backend/yolo_dataset_fixed")  # Новая папка!

# Очищаем старый датасет и создаем новый
if YOLO_DIR.exists():
    shutil.rmtree(YOLO_DIR)
    print("🗑️ Удален старый датасет")

YOLO_DIR.mkdir(exist_ok=True)
(YOLO_DIR / "images").mkdir(exist_ok=True)
(YOLO_DIR / "labels").mkdir(exist_ok=True)
(YOLO_DIR / "images/train").mkdir(exist_ok=True)
(YOLO_DIR / "images/val").mkdir(exist_ok=True)
(YOLO_DIR / "labels/train").mkdir(exist_ok=True)
(YOLO_DIR / "labels/val").mkdir(exist_ok=True)

# Классы
CLASSES = ["wall"]
class_to_id = {"wall": 0}

print(f"📁 Изображения: {IMAGES_DIR}")
print(f"📁 Разметки: {MARKUPS_DIR}")
print(f"📁 Новый YOLO датасет: {YOLO_DIR}")

# Функция для поиска изображения
def find_image(project_id, page_num):
    project_image_dir = IMAGES_DIR / project_id
    
    if not project_image_dir.exists():
        return None
    
    possible_patterns = [
        f"page_{page_num:03d}.jpg",
        f"page_{page_num}.jpg",
        f"page_{page_num:03d}.png",
        f"page_{page_num}.png",
    ]
    
    for pattern in possible_patterns:
        image_path = project_image_dir / pattern
        if image_path.exists():
            return image_path
    
    all_images = list(project_image_dir.glob("*.jpg")) + list(project_image_dir.glob("*.png"))
    if all_images and page_num <= len(all_images):
        all_images.sort()
        return all_images[page_num - 1]
    
    return None

# Функция конвертации
def convert_to_yolo_format(points, image_width, image_height):
    if not points or len(points) < 2:
        return []
    
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
    
    # Нормализуем
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

# Собираем все разметки
markup_files = list(MARKUPS_DIR.rglob("*.json"))
print(f"\n📊 Найдено файлов разметок: {len(markup_files)}")

# Словарь для хранения всех аннотаций по изображениям
image_annotations = defaultdict(list)  # ключ: (project_id, page_num), значение: список аннотаций
image_paths = {}  # ключ: (project_id, page_num), значение: путь к изображению
image_sizes = {}  # ключ: (project_id, page_num), значение: (width, height)

print("\n🔍 СБОР ВСЕХ РАЗМЕТОК...")

for markup_file in markup_files:
    try:
        with open(markup_file, "r", encoding="utf-8") as f:
            markup = json.load(f)
        
        project_id = markup.get("project_id", "")
        page_num = markup.get("page_num", 1)
        
        if not project_id:
            project_id = markup_file.parent.name
        
        key = (project_id, page_num)
        
        # Находим изображение (только один раз)
        if key not in image_paths:
            image_path = find_image(project_id, page_num)
            if image_path:
                image_paths[key] = image_path
                
                # Получаем размеры изображения
                img = cv2.imread(str(image_path))
                if img is not None:
                    height, width = img.shape[:2]
                    image_sizes[key] = (width, height)
                else:
                    print(f"   ❌ Не удалось прочитать {image_path}")
                    continue
            else:
                print(f"   ❌ Изображение не найдено для {project_id}, стр. {page_num}")
                continue
        
        # Получаем размеры изображения
        if key not in image_sizes:
            continue
            
        width, height = image_sizes[key]
        
        # Обрабатываем объекты
        if "objects" in markup:
            for obj in markup["objects"]:
                obj_type = obj.get("type", "")
                
                if obj_type == "wall" and "points" in obj:
                    points = obj["points"]
                    
                    yolo_coords = convert_to_yolo_format(points, width, height)
                    
                    if yolo_coords:
                        class_id = class_to_id.get(obj_type, 0)
                        line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in yolo_coords)
                        image_annotations[key].append(line)
                        
    except Exception as e:
        print(f"   ❌ Ошибка в {markup_file}: {e}")

print(f"\n📊 ОБРАБОТАНО:")
print(f"   Уникальных изображений: {len(image_paths)}")
print(f"   Всего аннотаций (стен): {sum(len(anns) for anns in image_annotations.values())}")

# Сохраняем данные
print("\n💾 СОХРАНЕНИЕ ДАННЫХ...")
image_counter = 0
annotation_counter = 0

for (project_id, page_num), annotations in image_annotations.items():
    if not annotations:
        continue
    
    key = (project_id, page_num)
    if key not in image_paths:
        continue
    
    image_path = image_paths[key]
    
    # Создаем имя файла
    image_name = f"{project_id}_p{page_num}.jpg"
    dest_image_path = YOLO_DIR / "images/train" / image_name
    
    # Копируем/конвертируем изображение
    img = cv2.imread(str(image_path))
    if img is None:
        continue
    
    if image_path.suffix.lower() in ['.png', '.jpeg']:
        cv2.imwrite(str(dest_image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        shutil.copy2(image_path, dest_image_path)
    
    # Сохраняем аннотации
    label_name = f"{project_id}_p{page_num}.txt"
    dest_label_path = YOLO_DIR / "labels/train" / label_name
    
    with open(dest_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations))
    
    image_counter += 1
    annotation_counter += len(annotations)
    
    print(f"   ✅ {image_name}: {len(annotations)} стен")

# Создаем dataset.yaml
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

print("\n" + "=" * 70)
print("📊 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
print("=" * 70)
print(f"✅ Изображений сохранено: {image_counter}")
print(f"✅ Всего аннотаций (стен): {annotation_counter}")

if annotation_counter > 0:
    print(f"🎯 Среднее количество стен на изображение: {annotation_counter / image_counter:.1f}")

print(f"\n📁 Папка с датасетом: {YOLO_DIR}")
print(f"📄 Файл конфигурации: {yaml_path}")

print("\n" + "=" * 70)
print("🚀 ГОТОВНОСТЬ К ОБУЧЕНИЮ:")
print("=" * 70)

if annotation_counter >= 20:
    print("✅ ОТЛИЧНО! 20+ стен - можно начинать обучение!")
elif annotation_counter >= 10:
    print("⚠️ НОРМАЛЬНО. 10+ стен - можно пробовать обучение.")
    print("   Для лучших результатов создайте еще разметок.")
else:
    print("❌ МАЛО. Нужно больше разметок стен.")
    print("   Минимум 10 стен рекомендуется.")

print("\nНажмите Enter для продолжения...")
input()
