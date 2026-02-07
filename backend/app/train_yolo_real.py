# train_yolo_real.py - РЕАЛЬНОЕ ОБУЧЕНИЕ YOLO
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import shutil

print("=" * 70)
print("🚀 ЗАПУСК РЕАЛЬНОГО ОБУЧЕНИЯ YOLO!")
print("=" * 70)

# Проверяем наличие GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"💻 Устройство для обучения: {device}")
print(f"🎮 CUDA доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Видеокарта: {torch.cuda.get_device_name(0)}")
    print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Пути
YOLO_DATASET = Path("C:/smet4ik/backend/yolo_dataset_fixed")
DATASET_YAML = YOLO_DATASET / "dataset.yaml"
MODELS_DIR = Path("C:/smet4ik/backend/app/ml_models")
MODELS_DIR.mkdir(exist_ok=True)

print(f"\n📁 Датасет: {YOLO_DATASET}")
print(f"📁 Папка для моделей: {MODELS_DIR}")

# Проверяем датасет
if not DATASET_YAML.exists():
    print(f"❌ Файл dataset.yaml не найден: {DATASET_YAML}")
    exit()

# Читаем конфигурацию датасета
with open(DATASET_YAML, 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f)

print(f"\n📊 КОНФИГУРАЦИЯ ДАТАСЕТА:")
print(f"   Путь: {dataset_config.get('path')}")
print(f"   Train: {dataset_config.get('train')}")
print(f"   Val: {dataset_config.get('val')}")
print(f"   Классы: {dataset_config.get('names', {})}")

# Проверяем количество изображений
train_images_dir = YOLO_DATASET / dataset_config.get('train', 'images/train')
train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))

print(f"\n📸 ИЗОБРАЖЕНИЯ ДЛЯ ОБУЧЕНИЯ: {len(train_images)}")
for img in train_images[:3]:  # Показываем первые 3
    print(f"   - {img.name}")

# Проверяем аннотации
train_labels_dir = YOLO_DATASET / "labels/train"  # Изменено на правильный путь
train_labels = list(train_labels_dir.glob("*.txt"))

print(f"\n📝 АННОТАЦИИ ДЛЯ ОБУЧЕНИЯ: {len(train_labels)}")

total_annotations = 0
for label in train_labels:
    with open(label, 'r', encoding='utf-8') as f:
        annotations = f.readlines()
        total_annotations += len(annotations)
    print(f"   - {label.name}: {len(annotations)} стен")

print(f"\n📈 ВСЕГО АННОТАЦИЙ (СТЕН): {total_annotations}")

# Спрашиваем подтверждение
print("\n⚠️  ВНИМАНИЕ: Обучение может занять от 10 минут до 1 часа")
print("   в зависимости от вашего компьютера.")
print("\n   Хотите начать обучение?")
response = input("   Введите 'ДА' для начала обучения: ")

if response.strip().upper() != 'ДА':
    print("\n❌ Обучение отменено пользователем")
    exit()

print("\n⏳ НАЧИНАЕМ ОБУЧЕНИЕ YOLO...")

try:
    # Загружаем предобученную модель YOLOv8
    print("\n1. Загрузка предобученной модели YOLOv8n-seg...")
    model = YOLO('yolov8n-seg.pt')  # Модель для сегментации
    
    # Настройки обучения
    print("2. Настройка параметров обучения...")
    
    # Обучаем модель
    print("3. Запуск обучения...")
    print("   Это может занять некоторое время. Пожалуйста, подождите...")
    print("   ⏳ Прогресс будет отображаться ниже:")
    print("   " + "-" * 50)
    
    results = model.train(
        data=str(DATASET_YAML),  # Конфигурационный файл датасета
        epochs=50,                # Количество эпох (циклов обучения)
        imgsz=640,                # Размер изображения
        batch=4,                  # Размер батча (меньше если мало памяти)
        device=device,            # Используем GPU или CPU
        workers=2,                # Количество потоков
        patience=10,              # Ранняя остановка если нет улучшений
        seed=42,                  # Для воспроизводимости
        project=str(MODELS_DIR),  # Куда сохранять результаты
        name='walls_yolo_v1',     # Имя эксперимента
        exist_ok=True,            # Перезаписывать если уже есть
        verbose=True              # Показывать подробный вывод
    )
    
    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("=" * 70)
    
    # Показываем результаты
    if hasattr(results, 'results_dict'):
        print("\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        for key, value in results.results_dict.items():
            print(f"   {key}: {value}")
    
    # Ищем созданные файлы модели
    trained_model_dir = MODELS_DIR / 'walls_yolo_v1'
    if trained_model_dir.exists():
        print(f"\n📁 Папка с обученной моделью: {trained_model_dir}")
        
        # Ищем лучшую модель
        best_model = trained_model_dir / 'weights' / 'best.pt'
        if best_model.exists():
            print(f"✅ Лучшая модель: {best_model}")
            
            # Копируем лучшую модель в основную папку для удобства
            dest_best = MODELS_DIR / 'best_walls_yolo.pt'
            shutil.copy2(best_model, dest_best)
            print(f"💾 Скопирована как: {dest_best}")
            
            # Показываем размер
            size_mb = dest_best.stat().st_size / (1024 * 1024)
            print(f"📦 Размер модели: {size_mb:.1f} MB")
        
        # Показываем все файлы в папке
        print(f"\n📋 Содержимое папки {trained_model_dir.name}:")
        for item in trained_model_dir.iterdir():
            if item.is_file():
                size_kb = item.stat().st_size / 1024
                print(f"   📄 {item.name} ({size_kb:.1f} KB)")
            elif item.is_dir():
                print(f"   📁 {item.name}/")
    
    print("\n🎉 МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
    print("   Теперь вы можете использовать её для обнаружения стен!")
    
    print("\n🚀 ДАЛЬНЕЙШИЕ ШАГИ:")
    print("   1. Загрузите новые чертежи через интерфейс")
    print("   2. Используйте /cv-dashboard/ для тестирования")
    print("   3. Модель автоматически будет использовать обученные веса")
    
except Exception as e:
    print(f"\n❌ ОШИБКА ПРИ ОБУЧЕНИИ: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n⚠️  ВОЗМОЖНЫЕ ПРИЧИНЫ:")
    print("   1. Недостаточно памяти (уменьшите batch в коде)")
    print("   2. Проблемы с датасетом (проверьте dataset.yaml)")
    print("   3. Библиотеки не установлены (pip install ultralytics)")

print("\nНажмите Enter для завершения...")
input()
