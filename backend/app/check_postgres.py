# check_postgres.py
print("=== Проверка подключения к PostgreSQL ===")

try:
    from database import db
    print("✅ Модуль database загружен")
    
    # Пробуем создать тестовую запись
    test_markup = {
        "project_id": "test_postgres",
        "page_num": 1,
        "objects": [{"type": "wall", "points": [{"x": 100, "y": 100}]}]
    }
    
    markup_id = db.save_markup("test_postgres", 1, test_markup)
    print(f"✅ Тестовая запись создана, ID: {markup_id}")
    
    # Проверяем статистику
    stats = db.get_training_statistics()
    print(f"✅ Статистика: {stats['training_markups']} разметок")
    
    print("=== PostgreSQL работает корректно! ===")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()