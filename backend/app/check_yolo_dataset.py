# check_yolo_dataset.py - Проверка YOLO датасета
import os
from pathlib import Path
import random

print("=" * 60)
print("🔍 ПРОВЕРКА СОЗДАННОГО YOLO ДАТАСЕТА")
print("=" * 60)

YOLO_DIR = Path("C:/smet4ik/backend/yolo_dataset")

if not YOLO_DIR.exists():
    print("❌ Папка yolo_dataset не найдена!")
    exit()

print(f"📁 YOLO датасет: {YOLO_DIR}")

# Проверяем структуру
print("\n📂 Структура папок:")
for root, dirs, files in os.walk(YOLO_DIR):
    level = root.replace(str(YOLO_DIR), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Показываем первые 5 файлов
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... и еще {len(files) - 5} файлов')

# Проверяем изображения
train_images = list((YOLO_DIR / "images/train").glob("*.jpg"))
val_images = list((YOLO_DIR / "images/val").glob("*.jpg"))

print(f"\n📊 СТАТИСТИКА:")
print(f"   Изображений для обучения: {len(train_images)}")
print(f"   Изображений для валидации: {len(val_images)}")

# Проверяем аннотации
train_labels = list((YOLO_DIR / "labels/train").glob("*.txt"))
val_labels = list((YOLO_DIR / "labels/val").glob("*.txt"))

print(f"   Аннотаций для обучения: {len(train_labels)}")
print(f"   Аннотаций для валидации: {len(val_labels)}")

# Считаем общее количество аннотаций (стен)
total_annotations = 0
print("\n📝 СОДЕРЖИМОЕ АННОТАЦИЙ:")

if train_labels:
    print("\n   Аннотации для обучения:")
    for label_file in train_labels[:3]:  # Показываем первые 3
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"   - {label_file.name}: {len(lines)} стен")
            total_annotations += len(lines)
            
            # Показываем первую аннотацию как пример
            if lines:
                parts = lines[0].strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    coords = parts[1:5]
                    print(f"     Пример: класс {class_id}, координаты {coords}")

# Проверяем dataset.yaml
yaml_file = YOLO_DIR / "dataset.yaml"
if yaml_file.exists():
    print(f"\n📄 ФАЙЛ КОНФИГУРАЦИИ ({yaml_file}):")
    with open(yaml_file, "r", encoding="utf-8") as f:
        print(f"\n{f.read()}")

print(f"\n📈 ИТОГО:")
print(f"   Всего изображений: {len(train_images) + len(val_images)}")
print(f"   Всего аннотаций (стен): {total_annotations}")

if total_annotations > 0:
    print(f"\n🎯 СРЕДНЕЕ КОЛИЧЕСТВО СТЕН НА ИЗОБРАЖЕНИЕ: {total_annotations / max(1, len(train_images)):.1f}")

print("\n" + "=" * 60)
print("🚀 ГОТОВНОСТЬ К ОБУЧЕНИЮ:")
print("=" * 60)

if total_annotations >= 10:
    print("✅ ОТЛИЧНО! Можно начинать обучение YOLO!")
    print("   Рекомендация: 10+ аннотаций достаточно для начала")
elif total_annotations >= 5:
    print("⚠️ МАЛОВАТО. Минимум 10 аннотаций рекомендуется.")
    print("   Создайте еще 5-10 разметок через /marker/")
else:
    print("❌ НЕДОСТАТОЧНО. Нужно больше разметок.")

print("\nНажмите Enter для продолжения...")
input()
