# debug_ocr.py
print("=== Отладка OCR сохранения ===")

import os
from pathlib import Path
from database import db
from ocr_processor import ocr_processor

# Берем последний проект
base_dir = Path("C:/smet4ik/backend/uploaded_pdfs")
projects = list(base_dir.iterdir())

if not projects:
    print("❌ Нет проектов")
    exit()

latest_project = projects[-1]  # Последний проект
print(f"Проект: {latest_project.name}")

# 1. Проверяем метаданные
metadata_file = latest_project / "metadata.json"
if metadata_file.exists():
    import json
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"1. Метаданные: OCR обработан = {metadata.get('ocr_processed')}")
    print(f"   OCR в базе = {metadata.get('ocr_saved_to_db', 'не указано')}")
    
    # 2. Проверяем изображения
    images_dir = Path("C:/smet4ik/backend/processed_images") / latest_project.name
    if images_dir.exists():
        images = list(images_dir.glob("*.jpg"))
        print(f"2. Изображений: {len(images)}")
        
        if images:
            # 3. Тестируем OCR на первой странице
            print(f"3. Тестируем OCR на {images[0].name}...")
            result = ocr_processor.analyze_page(images[0])
            print(f"   Найдено размеров: {result['measurements_count']}")
            print(f"   Ключевые слова: {result['keywords']}")
            
            # 4. Пробуем сохранить в базу
            print("4. Сохраняем в базу...")
            try:
                success = db.save_ocr_data(latest_project.name, 1, result)
                print(f"   Результат: {'✅ Успешно' if success else '❌ Ошибка'}")
            except Exception as e:
                print(f"   ❌ Ошибка сохранения: {e}")
    else:
        print("❌ Нет папки с изображениями")
else:
    print("❌ Нет metadata.json")

print("\n=== Завершено ===")