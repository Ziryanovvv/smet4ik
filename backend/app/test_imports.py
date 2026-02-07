# test_imports.py - Проверка всех импортов
import sys
import os

print("🔍 Проверка импортов Smet4ik AI Trainer")
print("=" * 50)

# Добавляем текущую директорию в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
print(f"📁 Текущая директория: {current_dir}")

try:
    print("1. Проверка FastAPI...")
    from fastapi import FastAPI
    print("✅ FastAPI импортирован")
except Exception as e:
    print(f"❌ Ошибка FastAPI: {e}")

try:
    print("2. Проверка базы данных...")
    # Пробуем импорт напрямую
    import database
    from database import db
    print("✅ Database импортирована")
    print(f"   Файл: {database.__file__}")
except Exception as e:
    print(f"❌ Ошибка базы данных: {e}")
    # Показываем список файлов
    print(f"📁 Содержимое папки {current_dir}:")
    files = [f for f in os.listdir(current_dir) if f.endswith('.py')]
    for f in files:
        print(f"   - {f}")

try:
    print("3. Проверка ML модели...")
    import ml_model
    from ml_model import wall_model
    print("✅ ML модель импортирована")
    print(f"   Файл: {ml_model.__file__}")
except Exception as e:
    print(f"❌ Ошибка ML модели: {e}")

try:
    print("4. Проверка CV модели...")
    import cv_model
    from cv_model import cv_model as cv_model_instance
    print("✅ CV модель импортирована")
    print(f"   Файл: {cv_model.__file__}")
except Exception as e:
    print(f"❌ Ошибка CV модели: {e}")

try:
    print("5. Проверка OCR процессора...")
    import ocr_processor
    from ocr_processor import ocr_processor as ocr_instance
    print("✅ OCR процессор импортирован")
    print(f"   Файл: {ocr_processor.__file__}")
except Exception as e:
    print(f"❌ Ошибка OCR процессора: {e}")

try:
    print("6. Проверка роутеров...")
    from app.routes import cv_router, markup_router, ml_router, ocr_router, upload_router
    print("✅ Все роутеры импортированы")
except Exception as e:
    print(f"❌ Ошибка роутеров: {e}")

try:
    print("7. Проверка main приложения...")
    from app.main import app
    print("✅ Приложение импортировано")
    print(f"   Файл: {app.__module__}")
except Exception as e:
    print(f"❌ Ошибка приложения: {e}")

print("=" * 50)
print("✅ Проверка импортов завершена!")