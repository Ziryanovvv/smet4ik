# train_yolo_simple.py - Простой скрипт обучения YOLO
import os
from pathlib import Path
from ultralytics import YOLO

print("=" * 60)
print("🚀 ЗАПУСК ПРОСТОГО ОБУЧЕНИЯ YOLO")
print("=" * 60)

# Пути
YOLO_DATASET = Path("C:/smet4ik/backend/yolo_dataset_fixed")
DATASET_YAML = YOLO_DATASET / "dataset.yaml"

print(f"📁 Датасет: {YOLO_DATASET}")
print(f"📄 Конфигурация: {DATASET_YAML}")

# Проверяем
if not DATASET_YAML.exists():
    print("❌ Файл dataset.yaml не найден!")
    exit()

# Проверяем папку валидации
val_dir = YOLO_DATASET / "images/val"
val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))

if len(val_images) == 0:
    print("❌ Нет изображений для валидации!")
    print("   Запустите сначала fix_dataset_for_training.py")
    exit()

print(f"\n✅ Изображений для валидации: {len(val_images)}")

# Загружаем модель
print("\n⏳ Загружаю модель YOLOv8n...")
model = YOLO('yolov8n.pt')  # Более простая модель для детекции (не сегментация)

# Настраиваем обучение
print("⏳ Настраиваю обучение...")

try:
    # Обучаем модель с минимальными настройками
    print("⏳ Начинаю обучение... (это займет 5-15 минут)")
    print("   Пожалуйста, подождите...\n")
    
    results = model.train(
        data=str(DATASET_YAML),  # Файл конфигурации
        epochs=30,                # Меньше эпох для быстрого теста
        imgsz=640,                # Размер изображения
        batch=2,                  # Маленький батч для CPU
        device='cpu',             # Используем CPU
        workers=1,                # Один поток
        patience=5,               # Ранняя остановка
        project='ml_models',      # Куда сохранять
        name='walls_trained',     # Имя модели
        exist_ok=True,            # Перезаписывать
        verbose=True              # Показывать прогресс
    )
    
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("=" * 60)
    
    # Проверяем созданные файлы
    model_dir = Path("ml_models/walls_trained")
    if model_dir.exists():
        print(f"\n📁 Папка с моделью: {model_dir}")
        
        # Ищем лучшую модель
        best_model = model_dir / 'weights' / 'best.pt'
        if best_model.exists():
            size_mb = best_model.stat().st_size / (1024 * 1024)
            print(f"✅ Лучшая модель: {best_model} ({size_mb:.1f} MB)")
            
            # Копируем для удобства
            import shutil
            simple_path = Path("ml_models/best_walls.pt")
            shutil.copy2(best_model, simple_path)
            print(f"💾 Скопирована как: {simple_path}")
        
        print("\n🎉 МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("   Теперь можете тестировать её в интерфейсе!")
        
except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")
    print("\n⚠️  Попробуйте:")
    print("   1. Убедиться что папка val не пустая")
    print("   2. Перезапустить fix_dataset_for_training.py")
    print("   3. Использовать yolov8n.pt вместо yolov8n-seg.pt")

print("\nНажмите Enter для завершения...")
input()
