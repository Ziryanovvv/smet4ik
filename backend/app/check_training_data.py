# check_training_data.py - Проверка данных для обучения
import json
import os
from pathlib import Path

print("=" * 60)
print("🔍 ПРОВЕРКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
print("=" * 60)

# 1. Проверяем разметки в папке markups
markups_dir = Path("C:/smet4ik/backend/app/markups")
print("1. Папка с разметками:", markups_dir)

if markups_dir.exists():
    all_markups = []
    
    # Ищем все JSON файлы
    for json_file in markups_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                markup = json.load(f)
            
            # Считаем объекты
            wall_count = 0
            window_count = 0
            door_count = 0
            
            if 'objects' in markup:
                for obj in markup['objects']:
                    if obj.get('type') == 'wall':
                        wall_count += 1
                    elif obj.get('type') == 'window':
                        window_count += 1
                    elif obj.get('type') == 'door':
                        door_count += 1
            
            all_markups.append({
                'file': str(json_file.relative_to(markups_dir)),
                'walls': wall_count,
                'windows': window_count,
                'doors': door_count,
                'total_objects': wall_count + window_count + door_count
            })
            
        except Exception as e:
            print(f"❌ Ошибка чтения {json_file}: {e}")
    
    print(f"   Найдено файлов разметок: {len(all_markups)}")
    
    if all_markups:
        total_walls = sum(m['walls'] for m in all_markups)
        total_windows = sum(m['windows'] for m in all_markups)
        total_doors = sum(m['doors'] for m in all_markups)
        
        print(f"   Всего стен: {total_walls}")
        print(f"   Всего окон: {total_windows}")
        print(f"   Всего дверей: {total_doors}")
        print(f"   Всего объектов: {total_walls + total_windows + total_doors}")
        
        # Показываем первые 5 файлов
        print("\n   Первые 5 разметок:")
        for i, markup in enumerate(all_markups[:5]):
            print(f"   {i+1}. {markup['file']}")
            print(f"      Стен: {markup['walls']}, Окон: {markup['windows']}, Дверей: {markup['doors']}")
    else:
        print("   ❌ Нет файлов разметок!")
        
else:
    print("   ❌ Папка markups не существует!")

# 2. Проверяем базу данных
print("\n2. База данных PostgreSQL:")
try:
    from database import db
    
    stats = db.get_training_statistics()
    print(f"   Всего разметок в БД: {stats['total_markups']}")
    print(f"   Для обучения: {stats['training_markups']}")
    print(f"   Проектов: {stats['projects_count']}")
    print(f"   OCR страниц: {stats['ocr_pages_processed']}")
    
except Exception as e:
    print(f"   ❌ Ошибка подключения к БД: {e}")

# 3. Проверяем папку с моделями
print("\n3. Папка с моделями (ml_models):")
ml_models_dir = Path("C:/smet4ik/backend/app/ml_models")
if ml_models_dir.exists():
    files = list(ml_models_dir.glob("*"))
    print(f"   Файлов в папке: {len(files)}")
    
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")
else:
    print("   ❌ Папка ml_models не существует!")

# 4. Проверяем изображения
print("\n4. Папка с изображениями (processed_images):")
proc_images_dir = Path("C:/smet4ik/backend/app/processed_images")
if proc_images_dir.exists():
    project_folders = list(proc_images_dir.iterdir())
    print(f"   Папок проектов: {len(project_folders)}")
    
    total_images = 0
    for project in project_folders:
        if project.is_dir():
            images = list(project.glob("*.jpg")) + list(project.glob("*.png"))
            total_images += len(images)
    
    print(f"   Всего изображений: {total_images}")
    
    if project_folders:
        print(f"   Первые 3 проекта:")
        for project in project_folders[:3]:
            if project.is_dir():
                images = list(project.glob("*.jpg")) + list(project.glob("*.png"))
                print(f"   - {project.name}: {len(images)} изображений")
else:
    print("   ❌ Папка processed_images не существует!")

print("\n" + "=" * 60)
print("📊 ИТОГ:")
print("=" * 60)

# Определяем состояние системы
if 'total_walls' in locals() and total_walls >= 3:
    print("✅ Есть достаточно данных для обучения (3+ стен)")
    print("   Рекомендация: Можно начинать обучение")
else:
    print("❌ Недостаточно данных для обучения")
    print("   Рекомендация: Создайте больше разметок через интерфейс /marker/")

print("\nСледующий шаг: будем готовить данные для реального обучения YOLO")
print("Нажмите Enter для продолжения...")
input()