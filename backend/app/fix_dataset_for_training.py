# fix_dataset_for_training.py - Исправляем датасет для обучения
import os
import shutil
import random
from pathlib import Path

print("=" * 60)
print("🔧 ИСПРАВЛЕНИЕ ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ YOLO")
print("=" * 60)

YOLO_DATASET = Path("C:/smet4ik/backend/yolo_dataset_fixed")

print(f"📁 Датасет: {YOLO_DATASET}")

# Проверяем существование
if not YOLO_DATASET.exists():
    print("❌ Датасет не найден!")
    exit()

# Пути
train_images_dir = YOLO_DATASET / "images/train"
train_labels_dir = YOLO_DATASET / "labels/train"
val_images_dir = YOLO_DATASET / "images/val"
val_labels_dir = YOLO_DATASET / "labels/val"

# Создаем папки если их нет
val_images_dir.mkdir(parents=True, exist_ok=True)
val_labels_dir.mkdir(parents=True, exist_ok=True)

# Получаем список изображений для обучения
train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
print(f"\n📸 Изображений для обучения: {len(train_images)}")

if len(train_images) < 2:
    print("❌ Нужно минимум 2 изображения для обучения!")
    exit()

# Если у нас только 2 изображения, берем одно для валидации
if len(train_images) == 2:
    # Первое изображение - для валидации, второе - для обучения
    val_image = train_images[0]
    val_label = train_labels_dir / f"{val_image.stem}.txt"
    
    print(f"\n📊 Только 2 изображения. Создаем валидацию:")
    print(f"   ✅ Валидация: {val_image.name}")
    print(f"   ✅ Обучение: {train_images[1].name}")
    
    # Перемещаем изображение для валидации
    shutil.move(str(val_image), str(val_images_dir / val_image.name))
    
    # Перемещаем соответствующую аннотацию
    if val_label.exists():
        shutil.move(str(val_label), str(val_labels_dir / val_label.name))
    
else:
    # Если больше 2 изображений, берем 20% для валидации
    val_count = max(1, int(len(train_images) * 0.2))
    val_indices = random.sample(range(len(train_images)), val_count)
    
    print(f"\n📊 Создаем валидацию ({val_count} из {len(train_images)} изображений):")
    
    for i in val_indices:
        val_image = train_images[i]
        val_label = train_labels_dir / f"{val_image.stem}.txt"
        
        print(f"   ✅ Валидация: {val_image.name}")
        
        # Перемещаем изображение для валидации
        shutil.move(str(val_image), str(val_images_dir / val_image.name))
        
        # Перемещаем соответствующую аннотацию
        if val_label.exists():
            shutil.move(str(val_label), str(val_labels_dir / val_label.name))

# Обновляем счетчики
train_images_after = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
val_images_after = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))

train_labels_after = list(train_labels_dir.glob("*.txt"))
val_labels_after = list(val_labels_dir.glob("*.txt"))

print(f"\n📊 НОВАЯ СТАТИСТИКА:")
print(f"   Обучение: {len(train_images_after)} изображений, {len(train_labels_after)} аннотаций")
print(f"   Валидация: {len(val_images_after)} изображений, {len(val_labels_after)} аннотаций")

# Обновляем dataset.yaml
yaml_file = YOLO_DATASET / "dataset.yaml"
if yaml_file.exists():
    print(f"\n📄 Обновляю файл конфигурации: {yaml_file}")
    
    yaml_content = f"""# YOLO Dataset Configuration
path: {YOLO_DATASET}  # dataset root dir
train: images/train  # train images
val: images/val  # val images

# Classes
names:
  0: wall
"""
    
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print("✅ Файл конфигурации обновлен!")

print("\n" + "=" * 60)
print("🎉 ДАТАСЕТ ИСПРАВЛЕН!")
print("=" * 60)

if len(val_images_after) > 0:
    print("✅ Теперь есть изображения для валидации")
    print(f"   Можно начинать обучение!")
else:
    print("❌ Все еще нет изображений для валидации")
    print("   Что-то пошло не так...")

print("\nНажмите Enter для продолжения...")
input()
