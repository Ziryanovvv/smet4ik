# check_db.py - Проверка подключения к базе данных
print("🔍 Проверка подключения к PostgreSQL...")

try:
    from database import db
    print("✅ Модуль database загружен")
    
    # Проверяем статистику
    stats = db.get_training_statistics()
    print(f"✅ Статистика из базы:")
    print(f"   Всего разметок: {stats.get('total_markups', 0)}")
    print(f"   Для обучения: {stats.get('training_markups', 0)}")
    print(f"   Проектов: {stats.get('projects_count', 0)}")
    print(f"   OCR страниц: {stats.get('ocr_pages_processed', 0)}")
    
    print("✅ PostgreSQL работает корректно!")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()