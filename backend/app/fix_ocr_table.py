# fix_ocr_table.py
print("=== Исправляем таблицу OCR ===")

try:
    import psycopg2
    
    # Подключаемся к базе
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        database='smet4ik_db',
        user='postgres',
        password='123'
    )
    cursor = conn.cursor()
    
    print("1. Проверяем таблицу ocr_data...")
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'ocr_data'
    """)
    
    columns = cursor.fetchall()
    print(f"   Колонки в таблице: {len(columns)}")
    for col in columns:
        print(f"   - {col[0]}: {col[1]}")
    
    print("\n2. Добавляем ограничение UNIQUE...")
    cursor.execute("""
        ALTER TABLE ocr_data 
        ADD CONSTRAINT ocr_data_unique UNIQUE (project_id, page_num)
    """)
    conn.commit()
    print("✅ Ограничение UNIQUE добавлено!")
    
    print("\n3. Проверяем...")
    cursor.execute("""
        SELECT COUNT(*) FROM ocr_data
    """)
    count = cursor.fetchone()[0]
    print(f"   Записей в таблице: {count}")
    
    conn.close()
    print("✅ База исправлена!")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")

print("\n=== Завершено ===")