import psycopg2
from psycopg2 import pool
import json
from pathlib import Path
from datetime import datetime
import os
import threading
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

class Database:
    def __init__(self):
        # Читаем параметры подключения из .env файла
        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'smet4ik_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '123')
        }
        
        self.lock = threading.Lock()
        self.connection_pool = None
        self.init_database()
        self.init_connection_pool()
        
        # Выводим информацию о подключении (без пароля)
        safe_params = self.db_params.copy()
        safe_params['password'] = '***'
        print(f"✅ Подключение к PostgreSQL: {safe_params}")
    
    def init_connection_pool(self):
        """Инициализация пула соединений"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20, **self.db_params
            )
            print("✅ Пул соединений PostgreSQL инициализирован")
        except Exception as e:
            print(f"❌ Ошибка подключения к PostgreSQL: {e}")
            print("Проверьте настройки в .env файле")
            raise
    
    def get_connection(self):
        """Получение соединения из пула"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Возврат соединения в пул"""
        self.connection_pool.putconn(conn)
    
    def init_database(self):
        """Инициализация базы данных"""
        conn = None
        try:
            # Временное подключение для создания таблиц
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Таблица проектов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    project_id TEXT UNIQUE NOT NULL,
                    original_filename TEXT NOT NULL,
                    total_pages INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица разметок
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS markups (
                    id SERIAL PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    markup_data TEXT NOT NULL,
                    is_training BOOLEAN DEFAULT TRUE,
                    accuracy REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            ''')
            
            # Таблица обученных моделей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    accuracy REAL DEFAULT 0.0,
                    training_samples INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица предсказаний ИИ
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    prediction_data TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    reviewed BOOLEAN DEFAULT FALSE,
                    correct BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # НОВАЯ ТАБЛИЦА: OCR данные
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_data (
                    id SERIAL PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    ocr_text TEXT,
                    measurements JSONB,
                    keywords JSONB,
                    measurements_count INTEGER DEFAULT 0,
                    has_architectural_data BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            ''')
            
            conn.commit()
            print("✅ Таблицы PostgreSQL созданы/проверены")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации БД: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def save_ocr_data(self, project_id, page_num, ocr_result):
        """Сохранение OCR данных в базу"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ocr_data 
                (project_id, page_num, ocr_text, measurements, keywords, measurements_count, has_architectural_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id, page_num) 
                DO UPDATE SET 
                    ocr_text = EXCLUDED.ocr_text,
                    measurements = EXCLUDED.measurements,
                    keywords = EXCLUDED.keywords,
                    measurements_count = EXCLUDED.measurements_count,
                    has_architectural_data = EXCLUDED.has_architectural_data,
                    created_at = CURRENT_TIMESTAMP
            ''', (
                project_id, 
                page_num,
                ocr_result.get('text_preview', ''),
                json.dumps(ocr_result.get('measurements', [])),
                json.dumps(ocr_result.get('keywords', [])),
                ocr_result.get('measurements_count', 0),
                ocr_result.get('has_architectural_data', False)
            ))
            
            conn.commit()
            print(f"✅ OCR данные сохранены: проект {project_id}, стр. {page_num}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения OCR данных: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_ocr_data(self, project_id, page_num=None):
        """Получение OCR данных"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if page_num:
                cursor.execute('''
                    SELECT * FROM ocr_data 
                    WHERE project_id = %s AND page_num = %s
                ''', (project_id, page_num))
            else:
                cursor.execute('''
                    SELECT * FROM ocr_data 
                    WHERE project_id = %s
                    ORDER BY page_num
                ''', (project_id,))
            
            rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            results = []
            for row in rows:
                result = dict(zip(column_names, row))
                # Парсим JSON поля
                if result.get('measurements'):
                    result['measurements'] = json.loads(result['measurements'])
                if result.get('keywords'):
                    result['keywords'] = json.loads(result['keywords'])
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка получения OCR данных: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_markup(self, project_id, page_num, markup_data, is_training=True):
        """Сохранение разметки в базу данных для обучения"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Сохраняем проект если его нет
            cursor.execute('''
                INSERT INTO projects (project_id, original_filename, total_pages)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id) DO NOTHING
            ''', (project_id, markup_data.get('original_filename', 'unknown'), 1))
            
            # Проверяем, есть ли уже разметка для этой страницы
            cursor.execute('''
                SELECT id FROM markups 
                WHERE project_id = %s AND page_num = %s
            ''', (project_id, page_num))
            
            existing = cursor.fetchone()
            
            # Добавляем OCR данные в разметку
            markup_with_ocr = {
                **markup_data,
                "ocr_data_added": datetime.now().isoformat(),
                "ocr_version": "1.0"
            }
            
            if existing:
                # Обновляем существующую
                cursor.execute('''
                    UPDATE markups 
                    SET markup_data = %s, is_training = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                ''', (json.dumps(markup_with_ocr), is_training, existing[0]))
                markup_id = existing[0]
            else:
                # Сохраняем новую разметку
                cursor.execute('''
                    INSERT INTO markups (project_id, page_num, markup_data, is_training)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                ''', (project_id, page_num, json.dumps(markup_with_ocr), is_training))
                markup_id = cursor.fetchone()[0]
            
            conn.commit()
            
            # Сохраняем в таблицу predictions как проверенный пример
            cursor.execute('''
                INSERT INTO predictions (project_id, page_num, prediction_data, confidence, reviewed, correct)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            ''', (project_id, page_num, json.dumps(markup_with_ocr), 1.0, True, True))
            
            conn.commit()
            
            print(f"✅ Разметка сохранена в PostgreSQL (ID: {markup_id}) для обучения: {is_training}")
            return markup_id
            
        except Exception as e:
            print(f"❌ Ошибка сохранения в PostgreSQL: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_markup_to_file(self, project_id, page_num, markup_data):
        """Сохранение разметки в файл в структурированной директории"""
        # Создаем директорию для разметок (абсолютный путь)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        markups_dir = Path(base_dir) / "markups"
        markups_dir.mkdir(exist_ok=True)
        
        # Создаем поддиректорию для проекта
        project_markups_dir = markups_dir / project_id
        project_markups_dir.mkdir(exist_ok=True)
        
        # Генерируем уникальный ID для разметки
        markup_id = f"{project_id}_p{page_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Сохраняем файл
        markup_file = project_markups_dir / f"{markup_id}.json"
        
        # Добавляем метаданные и OCR данные
        markup_with_meta = {
            **markup_data,
            "markup_id": markup_id,
            "saved_at": datetime.now().isoformat(),
            "file_path": str(markup_file),
            "ocr_enhanced": True,
            "ocr_added_at": datetime.now().isoformat()
        }
        
        with open(markup_file, "w", encoding="utf-8") as f:
            json.dump(markup_with_meta, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Разметка сохранена в файл: {markup_file}")
        return markup_id, str(markup_file)
    
    def get_all_markups(self):
        """Получение всех сохраненных разметок"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        markups_dir = Path(base_dir) / "markups"
        
        if not markups_dir.exists():
            return []
        
        markups = []
        
        # Рекурсивно ищем все JSON файлы
        for json_file in markups_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    markup_data = json.load(f)
                    
                # Добавляем информацию о файле
                markup_info = {
                    "file_path": str(json_file),
                    "file_name": json_file.name,
                    "project_id": json_file.parent.name,
                    "markup_id": markup_data.get("markup_id", ""),
                    "created_at": markup_data.get("saved_at", ""),
                    "total_objects": markup_data.get("total_objects", 0),
                    "has_walls": any(obj.get("type") == "wall" for obj in markup_data.get("objects", [])),
                    "walls_count": sum(1 for obj in markup_data.get("objects", []) if obj.get("type") == "wall"),
                    "preview": f"{markup_data.get('total_objects', 0)} объектов",
                    "ocr_enhanced": markup_data.get("ocr_enhanced", False)
                }
                
                markups.append(markup_info)
                
            except Exception as e:
                print(f"Ошибка чтения файла {json_file}: {e}")
        
        # Сортируем по дате создания (новые сначала)
        markups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return markups
    
    def get_markup_by_id(self, markup_id):
        """Получение разметки по ID"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        markups_dir = Path(base_dir) / "markups"
        
        if not markups_dir.exists():
            return None
        
        # Ищем файл с таким ID
        for json_file in markups_dir.rglob("*.json"):
            if markup_id in json_file.stem:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Ошибка чтения файла {json_file}: {e}")
        
        return None
    
    def delete_markup(self, markup_id):
        """Удаление разметки по ID"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        markups_dir = Path(base_dir) / "markups"
        
        if not markups_dir.exists():
            return False
        
        # Ищем файл с таким ID
        for json_file in markups_dir.rglob("*.json"):
            if markup_id in json_file.stem:
                try:
                    json_file.unlink()  # Удаляем файл
                    print(f"✅ Разметка удалена: {json_file}")
                    return True
                except Exception as e:
                    print(f"Ошибка удаления файла {json_file}: {e}")
                    return False
        
        return False
    
    def get_markups_for_training(self, limit=100):
        """Получение разметок для обучения"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM markups 
                WHERE is_training = TRUE 
                ORDER BY created_at DESC 
                LIMIT %s
            ''', (limit,))
            
            rows = cursor.fetchall()
            
            # Получаем имена колонок
            column_names = [desc[0] for desc in cursor.description]
            
            markups = []
            for row in rows:
                markup = dict(zip(column_names, row))
                markup['markup_data'] = json.loads(markup['markup_data'])
                markups.append(markup)
            
            return markups
            
        except Exception as e:
            print(f"❌ Ошибка получения разметок для обучения: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_training_statistics(self):
        """Получение статистики по данным для обучения"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_markups,
                    SUM(CASE WHEN is_training = TRUE THEN 1 ELSE 0 END) as training_markups,
                    SUM(CASE WHEN is_training = FALSE THEN 1 ELSE 0 END) as validation_markups
                FROM markups
            ''')
            
            stats = cursor.fetchone()
            
            cursor.execute('''
                SELECT COUNT(DISTINCT project_id) as projects_count
                FROM markups
            ''')
            
            projects_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) as walls_count
                FROM markups
                WHERE markup_data LIKE '%%"type": "wall"%%'
            ''')
            
            walls_count = cursor.fetchone()[0]
            
            # OCR статистика
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_ocr_pages,
                    SUM(measurements_count) as total_measurements,
                    COUNT(CASE WHEN has_architectural_data = TRUE THEN 1 END) as pages_with_arch_data
                FROM ocr_data
            ''')
            
            ocr_stats = cursor.fetchone()
            
            # Также получаем статистику из файлов
            file_stats = self.get_all_markups()
            
            return {
                'total_markups': stats[0] if stats[0] else 0,
                'training_markups': stats[1] if stats[1] else 0,
                'validation_markups': stats[2] if stats[2] else 0,
                'projects_count': projects_count,
                'walls_count': walls_count,
                'file_markups_count': len(file_stats),
                'ocr_pages_processed': ocr_stats[0] if ocr_stats and ocr_stats[0] else 0,
                'total_measurements_found': ocr_stats[1] if ocr_stats and ocr_stats[1] else 0,
                'pages_with_architectural_data': ocr_stats[2] if ocr_stats and ocr_stats[2] else 0
            }
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return {
                'total_markups': 0,
                'training_markups': 0,
                'validation_markups': 0,
                'projects_count': 0,
                'walls_count': 0,
                'file_markups_count': 0,
                'ocr_pages_processed': 0,
                'total_measurements_found': 0,
                'pages_with_architectural_data': 0
            }
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_prediction(self, project_id, page_num, prediction_data, confidence):
        """Сохранение предсказания ИИ"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (project_id, page_num, prediction_data, confidence)
                VALUES (%s, %s, %s, %s)
            ''', (project_id, page_num, json.dumps(prediction_data), confidence))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения предсказания: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def update_prediction_review(self, prediction_id, is_correct):
        """Обновление статуса проверки предсказания"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE predictions 
                SET reviewed = TRUE, correct = %s
                WHERE id = %s
            ''', (is_correct, prediction_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обновления предсказания: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)

# Глобальный экземпляр базы данных
db = Database()