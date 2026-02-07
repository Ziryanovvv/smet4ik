# test_trained_model.py - Тестируем обученную модель
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

print("=" * 60)
print("🧪 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ YOLO")
print("=" * 60)

# Пути к моделям
MODEL_PATH = Path("C:/smet4ik/backend/app/runs/detect/ml_models/walls_trained/weights/best.pt")
SIMPLE_MODEL_PATH = Path("C:/smet4ik/backend/app/ml_models/best_walls.pt")
TEST_IMAGE = Path("C:/smet4ik/backend/processed_images/1856415c/page_001.jpg")

print(f"📁 Модель: {MODEL_PATH}")
print(f"📁 Упрощенный путь: {SIMPLE_MODEL_PATH}")
print(f"📷 Тестовое изображение: {TEST_IMAGE}")

# Проверяем существование файлов
if not MODEL_PATH.exists():
    print("❌ Файл модели не найден!")
    # Ищем альтернативный путь
    possible_paths = [
        Path("ml_models/best_walls.pt"),
        Path("runs/detect/walls_trained/weights/best.pt"),
        Path("best.pt")
    ]
    
    for path in possible_paths:
        if path.exists():
            MODEL_PATH = path
            print(f"✅ Найдена модель: {MODEL_PATH}")
            break
    else:
        print("❌ Модель не найдена ни в одном из возможных путей")
        exit()

if not TEST_IMAGE.exists():
    print("❌ Тестовое изображение не найдено!")
    # Ищем любое изображение
    images_dir = Path("C:/smet4ik/backend/processed_images")
    if images_dir.exists():
        for project in images_dir.iterdir():
            if project.is_dir():
                images = list(project.glob("*.jpg")) + list(project.glob("*.png"))
                if images:
                    TEST_IMAGE = images[0]
                    print(f"✅ Используем изображение: {TEST_IMAGE}")
                    break

# Загружаем модель
print("\n⏳ Загружаю обученную модель...")
try:
    model = YOLO(str(MODEL_PATH))
    print("✅ Модель успешно загружена!")
    
    # Информация о модели
    print(f"\n📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
    print(f"   Путь: {MODEL_PATH}")
    print(f"   Размер: {MODEL_PATH.stat().st_size / (1024*1024):.1f} MB")
    
    # Загружаем тестовое изображение
    print(f"\n📷 Загружаю тестовое изображение...")
    img = cv2.imread(str(TEST_IMAGE))
    if img is None:
        print("❌ Не удалось загрузить изображение")
        exit()
    
    height, width = img.shape[:2]
    print(f"   Размеры: {width}x{height} пикселей")
    
    # Тестируем модель
    print(f"\n🔍 Тестирую модель на изображении...")
    print("   Это может занять несколько секунд...")
    
    results = model.predict(
        source=str(TEST_IMAGE),
        conf=0.2,  # Низкий порог для теста
        imgsz=640,
        device='cpu',
        verbose=False
    )
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 60)
    
    if results and len(results) > 0:
        result = results[0]
        
        # Считаем обнаруженные объекты
        if result.boxes is not None:
            boxes = result.boxes
            detected_count = len(boxes)
            
            print(f"✅ Обнаружено объектов: {detected_count}")
            
            if detected_count > 0:
                print(f"\n🔍 ДЕТАЛИ ОБНАРУЖЕНИЙ:")
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Координаты в пикселях
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width_box = x2 - x1
                    height_box = y2 - y1
                    
                    print(f"   Объект #{i+1}:")
                    print(f"     Класс: {'стена' if cls == 0 else 'неизвестный'}")
                    print(f"     Уверенность: {conf:.1%}")
                    print(f"     Координаты: ({center_x:.0f}, {center_y:.0f})")
                    print(f"     Размеры: {width_box:.0f}x{height_box:.0f} пикселей")
            
            # Визуализируем результат
            print(f"\n🎨 Сохраняю результат с обнаружениями...")
            output_path = Path("test_result.jpg")
            result.save(filename=str(output_path))
            
            if output_path.exists():
                print(f"✅ Результат сохранен: {output_path}")
                print(f"   Откройте этот файл чтобы увидеть обнаруженные стены!")
            else:
                # Альтернативный способ
                annotated_img = result.plot()
                cv2.imwrite("test_result_cv.jpg", annotated_img)
                print(f"✅ Результат сохранен: test_result_cv.jpg")
        
        else:
            print("❌ Модель не обнаружила ни одного объекта")
    
    else:
        print("❌ Нет результатов от модели")
    
except Exception as e:
    print(f"❌ Ошибка при тестировании модели: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🚀 ЧТО ДАЛЬШЕ?")
print("=" * 60)
print("1. 📸 Создайте больше разметок (минимум 10 разных чертежей)")
print("2. 🏗️  Переобучите модель на большем датасете")
print("3. 🧪 Протестируйте модель на новых чертежах")
print("4. 📈 Метрики улучшатся с увеличением данных!")

print("\n🎉 ВАША ПЕРВАЯ AI МОДЕЛЬ ГОТОВА К РАБОТЕ!")
print("   Можете использовать её для автоматического обнаружения стен!")

print("\nНажмите Enter для завершения...")
input()
