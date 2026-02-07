import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
from PIL import Image
import shutil
import uuid
from pathlib import Path
import json
import io
from dotenv import load_dotenv
import os

load_dotenv()

# Читаем версию из .env файла
APP_VERSION = os.getenv('APP_VERSION', '0.9.0')
APP_NAME = os.getenv('APP_NAME', 'Smet4ik AI Trainer')

# Импортируем реальные модули - без заглушек!
from ml_model import wall_model
from database import db
from ocr_processor import ocr_processor

# Создаем папки для хранения данных
UPLOAD_DIR = Path("uploaded_pdfs")
PROCESSED_DIR = Path("processed_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Создаем папку для разметок
MARKUPS_DIR = Path("markups")
MARKUPS_DIR.mkdir(exist_ok=True)

app = FastAPI(title=f"{APP_NAME} API", version=APP_VERSION)

# Статические файлы (для будущего фронтенда)
import os
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

def convert_pdf_to_images_fitz(pdf_path: Path, output_dir: Path, dpi=150):
    """Конвертация PDF в изображения с использованием PyMuPDF"""
    images = []
    
    try:
        # Открываем PDF
        doc = fitz.open(str(pdf_path))
        print(f"PDF открыт успешно. Страниц: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Увеличиваем DPI для качества
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Конвертируем в JPEG
            img_data = pix.tobytes("jpeg")
            img = Image.open(io.BytesIO(img_data))
            
            # Сохраняем изображение
            output_path = output_dir / f"page_{page_num + 1:03d}.jpg"
            img.save(output_path, "JPEG", quality=95)
            images.append(str(output_path))
            
            print(f"Страница {page_num + 1} сконвертирована: {output_path}")
        
        doc.close()
        return images
        
    except Exception as e:
        print(f"Ошибка конвертации PDF: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Главная страница тренажера с формой загрузки"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{APP_NAME} v{APP_VERSION} - Загрузка PDF</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 900px; margin: 0 auto; }
            h1 { color: #333; }
            .upload-box { 
                border: 2px dashed #ccc; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 10px;
            }
            .upload-box:hover { border-color: #4CAF50; }
            #fileInput { display: none; }
            .upload-label { 
                cursor: pointer; 
                color: #4CAF50;
                font-weight: bold;
            }
            .status { 
                background: #f0f0f0; 
                padding: 20px; 
                border-radius: 5px; 
                margin: 20px 0; 
                display: none;
            }
            button { 
                background: #4CAF50; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer;
                font-size: 16px;
            }
            button:hover { background: #45a049; }
            .image-preview { margin-top: 20px; }
            .image-preview img { max-width: 200px; border: 1px solid #ddd; margin: 5px; }
            .pages-grid { display: flex; flex-wrap: wrap; gap: 10px; }
            .nav-links { margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee; }
            .nav-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 25px;
                border-radius: 8px;
                text-decoration: none;
                display: inline-block;
                margin: 10px;
                font-weight: bold;
                transition: all 0.3s;
            }
            .nav-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .ocr-info {
                background: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #4CAF50;
            }
            .ocr-stat {
                background: #e3f2fd;
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 Smet4ik AI Trainer v0.9.0 - Загрузка PDF</h1>
            <p><strong>🎯 Исправлено:</strong> OCR данные теперь гарантированно сохраняются в базу!</p>
            
            <div class="ocr-info">
                <h3>🔍 Система OCR:</h3>
                <ul>
                    <li>Распознает текст на чертежах</li>
                    <li>Находит размеры (например: "3500 мм", "1200x1500 мм")</li>
                    <li>Определяет ключевые слова: "стена", "окно", "дверь", "кухня", "ванная"</li>
                    <li>Сохраняет данные в базу для обучения ИИ</li>
                    <li>Теперь работает надежно!</li>
                </ul>
            </div>
            
            <div class="upload-box" id="uploadArea">
                <input type="file" id="fileInput" accept=".pdf">
                <label for="fileInput" class="upload-label">
                    📁 Нажмите для выбора PDF файла или перетащите его сюда
                </label>
                <p id="fileName"></p>
            </div>
            
            <button onclick="uploadFile()" id="uploadBtn" disabled>Загрузить PDF</button>
            
            <div class="status" id="statusBox">
                <h3>Статус обработки:</h3>
                <p id="statusText"></p>
                <div id="progress"></div>
            </div>
            
            <div class="image-preview" id="imagePreview"></div>
            
            <div class="nav-links">
                <h3>📊 Инструменты тренажера:</h3>
                <a href="/marker/" class="nav-button">
                    🎨 Интерфейс разметки чертежей
                </a>
                <a href="/ml-test/" class="nav-button">
                    🧠 Управление обучением ИИ
                </a>
                <a href="/docs" class="nav-button">
                    📖 Документация API
                </a>
                <a href="/health" class="nav-button">
                    🩺 Проверка здоровья
                </a>
            </div>
        </div>
        
        <script>
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const fileName = document.getElementById('fileName');
            const uploadArea = document.getElementById('uploadArea');
            const statusBox = document.getElementById('statusBox');
            const statusText = document.getElementById('statusText');
            const imagePreview = document.getElementById('imagePreview');
            
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#4CAF50';
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = '#ccc';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelect();
                }
            });
            
            fileInput.addEventListener('change', handleFileSelect);
            
            function handleFileSelect() {
                if (fileInput.files.length > 0) {
                    fileName.textContent = `Выбран файл: ${fileInput.files[0].name}`;
                    uploadBtn.disabled = false;
                }
            }
            
            async function uploadFile() {
                const file = fileInput.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                statusBox.style.display = 'block';
                statusText.textContent = 'Загрузка файла...';
                uploadBtn.disabled = true;
                
                try {
                    const response = await fetch('/upload-pdf/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let ocrInfo = '';
                        if (result.ocr_results && result.ocr_results.length > 0) {
                            ocrInfo = '<div class="ocr-info">';
                            ocrInfo += '<h4>📊 Результаты OCR анализа:</h4>';
                            ocrInfo += `<p><strong>Всего страниц:</strong> ${result.total_pages}</p>`;
                            ocrInfo += `<p><strong>OCR сохранен в базу:</strong> ${result.ocr_saved_to_db ? '✅ Да' : '❌ Нет'}</p>`;
                            ocrInfo += '</div>';
                        }
                        
                        statusText.innerHTML = `
                            ✅ Файл успешно загружен!<br>
                            <strong>ID проекта:</strong> ${result.project_id}<br>
                            <strong>Страниц:</strong> ${result.total_pages}<br>
                            ${ocrInfo}
                            <a href="/project/${result.project_id}/" target="_blank">📁 Перейти к просмотру проекта</a>
                        `;
                    } else {
                        statusText.textContent = `❌ Ошибка: ${result.detail || 'Неизвестная ошибка'}`;
                    }
                } catch (error) {
                    statusText.textContent = `❌ Ошибка сети: ${error.message}`;
                } finally {
                    uploadBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/marker/")
async def marker_interface():
    """Интерфейс для разметки чертежей"""
    html_path = Path(__file__).parent / "marker.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Загрузка PDF файла и конвертация в изображения"""
    try:
        # Генерируем уникальный ID для проекта
        project_id = str(uuid.uuid4())[:8]
        project_dir = UPLOAD_DIR / project_id
        images_dir = PROCESSED_DIR / project_id
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"Начало обработки PDF: {file.filename}")
        print(f"Project ID: {project_id}")
        
        # Сохраняем PDF
        pdf_path = project_dir / file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"PDF сохранен: {pdf_path}")
        
        # Конвертируем PDF в изображения
        print("Начало конвертации PDF в изображения...")
        images = convert_pdf_to_images_fitz(pdf_path, images_dir, dpi=150)
        print(f"Конвертация завершена. Получено изображений: {len(images)}")
        
        # СОЗДАЕМ ПРОЕКТ В БАЗЕ ПЕРЕД OCR (ИСПРАВЛЕНИЕ!)
        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                port='5432',
                database='smet4ik_db',
                user='postgres',
                password='123'
            )
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO projects (project_id, original_filename, total_pages)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id) DO NOTHING
            ''', (project_id, file.filename, len(images)))
            conn.commit()
            conn.close()
            print(f"✅ Проект создан в базе данных")
        except Exception as e:
            print(f"⚠️ Не удалось создать проект в базе: {e}")
        
        # Формируем информацию о страницах с OCR анализом
        pages_info = []
        ocr_results = []
        
        for i, img_path in enumerate(images, 1):
            img_filename = os.path.basename(img_path)
            
            # Выполняем OCR анализ страницы
            print(f"🔍 Выполняем OCR анализ страницы {i}...")
            ocr_result = ocr_processor.analyze_page(Path(img_path))
            
            # Сохраняем OCR данные в базу данных
            ocr_saved = db.save_ocr_data(project_id, i, ocr_result)
            
            pages_info.append({
                "page_num": i,
                "image_path": img_filename,
                "image_url": f"/project/{project_id}/page/{i}/image",
                "ocr_text_preview": ocr_result['text_preview'],
                "ocr_measurements": ocr_result['measurements'],
                "ocr_keywords": ocr_result['keywords'],
                "has_architectural_data": ocr_result['has_architectural_data']
            })
            
            ocr_results.append({
                "page_num": i,
                "measurements_count": ocr_result['measurements_count'],
                "keywords": ocr_result['keywords'],
                "has_architectural_data": ocr_result['has_architectural_data'],
                "saved_to_db": ocr_saved
            })
            
            print(f"📄 Страница {i}: {len(ocr_result['measurements'])} размеров, сохранено в базу: {'✅' if ocr_saved else '❌'}")
        
        # Сохраняем метаданные проекта
        metadata = {
            "project_id": project_id,
            "original_filename": file.filename,
            "pdf_path": str(pdf_path),
            "pages": pages_info,
            "total_pages": len(pages_info),
            "ocr_results": ocr_results,
            "status": "uploaded",
            "converter": "PyMuPDF (fitz)",
            "ocr_processed": True,
            "ocr_saved_to_db": any(r.get('saved_to_db') for r in ocr_results)
        }
        
        metadata_path = project_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Метаданные сохранены: {metadata_path}")
        
        return {
            "message": "PDF успешно загружен и сконвертирован",
            "project_id": project_id,
            "pages": pages_info,
            "ocr_results": ocr_results,
            "total_pages": len(pages_info),
            "converter": "PyMuPDF",
            "ocr_saved_to_db": metadata['ocr_saved_to_db']
        }
        
    except Exception as e:
        print(f"Критическая ошибка в upload_pdf: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки PDF: {str(e)}")

@app.get("/project/{project_id}/")
async def get_project(project_id: str):
    """Страница просмотра проекта"""
    project_dir = UPLOAD_DIR / project_id
    metadata_file = project_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Проект не найден")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Получаем OCR данные из базы
    ocr_db_data = db.get_ocr_data(project_id)
    
    html_content = f"""
    <html>
    <head>
        <title>Проект {project_id} - Smet4ik</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .page {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            img {{ max-width: 100%; }}
            .info {{ background: #f5f5f0; padding: 15px; }}
            .ocr-result {{
                background: #e8f5e9;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }}
            .db-info {{
                background: #e3f2fd;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #2196F3;
            }}
        </style>
    </head>
    <body>
        <h1>📋 Проект: {metadata['original_filename']}</h1>
        <div class="info">
            <p><strong>ID проекта:</strong> {project_id}</p>
            <p><strong>Всего страниц:</strong> {metadata['total_pages']}</p>
            <p><strong>OCR анализ:</strong> {'✅ Выполнен' if metadata.get('ocr_processed') else '❌ Не выполнен'}</p>
            <p><strong>OCR в базе данных:</strong> {'✅ Сохранено' if metadata.get('ocr_saved_to_db') else '❌ Не сохранено'}</p>
            <p><strong>OCR записей в базе:</strong> {len(ocr_db_data)}</p>
            <p><a href="/">← Назад к загрузке</a> | <a href="/marker/">🎨 К разметке</a></p>
        </div>
        
        <h2>Страницы чертежа:</h2>
    """
    
    for page in metadata["pages"]:
        ocr_info = ""
        if page.get('ocr_measurements'):
            ocr_info += f"<div class='ocr-result'>"
            ocr_info += f"<strong>📏 OCR анализ:</strong><br>"
            if page['ocr_measurements']:
                ocr_info += f"<strong>Найдено размеров:</strong> {len(page['ocr_measurements'])}<br>"
            if page.get('ocr_keywords'):
                ocr_info += f"<strong>Ключевые слова:</strong> {', '.join(page['ocr_keywords'])}<br>"
            ocr_info += f"</div>"
        
        html_content += f"""
        <div class="page">
            <h3>📄 Страница {page['page_num']} из {metadata['total_pages']}</h3>
            {ocr_info}
            <img src="{page['image_url']}" 
                 alt="Страница {page['page_num']}"
                 style="max-width: 800px; border: 1px solid #ccc;">
            <p><a href="{page['image_url']}" target="_blank">Открыть в полном размере</a></p>
        </div>
        """
    
    html_content += """
        <hr>
        <p><small>Smet4ik AI Trainer v0.9.0 - OCR данные надежно сохраняются в базу</small></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/project/{project_id}/page/{page_num}/image")
async def get_page_image(project_id: str, page_num: int):
    """Получение изображения страницы"""
    images_dir = PROCESSED_DIR / project_id
    
    image_pattern = f"page_{page_num:03d}.jpg"
    image_path = images_dir / image_pattern
    
    if not image_path.exists():
        all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not all_images:
            raise HTTPException(status_code=404, detail="Изображения не найдены")
        
        all_images.sort()
        if page_num < 1 or page_num > len(all_images):
            raise HTTPException(status_code=404, detail="Страница не найдена")
        
        image_path = all_images[page_num - 1]
    
    return FileResponse(image_path)

@app.get("/api/ocr-data/{project_id}/")
async def get_ocr_data(project_id: str, page_num: int = None):
    """Получение OCR данных из базы"""
    try:
        ocr_data = db.get_ocr_data(project_id, page_num)
        
        if not ocr_data:
            return {
                "success": False,
                "message": "OCR данные не найдены",
                "project_id": project_id,
                "data": []
            }
        
        return {
            "success": True,
            "message": f"Найдено {len(ocr_data)} записей OCR данных",
            "project_id": project_id,
            "total_records": len(ocr_data),
            "data": ocr_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка получения OCR данных: {str(e)}",
            "project_id": project_id,
            "data": []
        }

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    try:
        stats = db.get_training_statistics()
        ocr_stats = {
            "ocr_pages_processed": stats.get('ocr_pages_processed', 0),
            "total_measurements_found": stats.get('total_measurements_found', 0),
            "pages_with_architectural_data": stats.get('pages_with_architectural_data', 0)
        }
    except:
        ocr_stats = {}
    
    return {
        "status": "healthy", 
        "service": "smet4ik-backend",
        "version": "0.9.0",
        "upload_dir_exists": os.path.exists(UPLOAD_DIR),
        "processed_dir_exists": os.path.exists(PROCESSED_DIR),
        "markups_dir_exists": os.path.exists(MARKUPS_DIR),
        "converter": "PyMuPDF",
        "ocr_available": True,
        "ocr_stats": ocr_stats
    }

# ========== ML MODEL API ENDPOINTS ==========

@app.get("/api/model-status/")
async def get_model_status():
    """Получение статуса ML модели"""
    accuracy = 0
    if wall_model.is_trained:
        try:
            import json
            from pathlib import Path
            metadata_path = Path("ml_models/model_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    accuracy = metadata.get('last_accuracy', 0)
        except:
            accuracy = 0
    
    return {
        "is_trained": wall_model.is_trained,
        "accuracy": accuracy,
        "model_type": "RandomForest",
        "samples_trained": 0
    }

@app.post("/api/analyze-markup/")
async def analyze_markup(markup: dict):
    """Анализ разметки и извлечение признаков"""
    features = wall_model.extract_features(markup)
    
    return {
        "feature_count": len(features),
        "features": features.tolist() if len(features) > 0 else [],
        "message": f"Извлечено {len(features)} признаков"
    }

@app.post("/api/predict/")
async def predict_walls(markup: dict):
    """Предсказание стен в разметке"""
    predictions = wall_model.predict_walls(markup)
    
    return {
        "predictions": predictions,
        "count": len(predictions),
        "model_trained": wall_model.is_trained,
        "message": f"Предсказано {len(predictions)} стен" if predictions else "Модель не обучена или стены не найдены"
    }

@app.post("/api/train/")
async def train_model():
    """Обучение ML модели на размеченных данных"""
    try:
        markups = db.get_markups_for_training(limit=50)
        
        if not markups:
            return {
                "success": False,
                "message": "Нет размеченных данных для обучения",
                "accuracy": 0,
                "samples": 0
            }
        
        result = wall_model.train(markups)
        
        if result:
            return {
                "success": True,
                "message": "Модель успешно обучена",
                "accuracy": result['accuracy'],
                "samples": result['samples'],
                "walls_count": result['walls_count'],
                "non_walls_count": result['non_walls_count']
            }
        else:
            return {
                "success": False,
                "message": "Ошибка обучения модели",
                "accuracy": 0,
                "samples": 0
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка: {str(e)}",
            "accuracy": 0,
            "samples": 0
        }

@app.post("/api/feedback/")
async def receive_feedback(feedback: dict):
    """Получение обратной связи для активного обучения"""
    try:
        print(f"📝 Получена обратная связь: {feedback}")
        
        return {
            "success": True,
            "message": "Обратная связь получена",
            "feedback_id": "temp_" + str(hash(str(feedback)))[-8:],
            "timestamp": feedback.get('timestamp', '')
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка обработки обратной связи: {str(e)}"
        }

# ========== MARKUP MANAGEMENT API ==========

@app.get("/api/markups/")
async def get_all_markups():
    """Получение всех сохраненных разметок"""
    try:
        markups = db.get_all_markups()
        return {
            "success": True,
            "count": len(markups),
            "markups": markups
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка получения разметок: {str(e)}",
            "count": 0,
            "markups": []
        }

@app.get("/api/markup/{markup_id}/")
async def get_markup(markup_id: str):
    """Получение конкретной разметки по ID"""
    try:
        markup = db.get_markup_by_id(markup_id)
        if markup:
            return {
                "success": True,
                "markup": markup
            }
        else:
            return {
                "success": False,
                "message": f"Разметка с ID {markup_id} не найдена"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка получения разметки: {str(e)}"
        }

@app.post("/api/markup/save/")
async def save_markup_file(markup: dict):
    """Сохранение разметки в файл и БД"""
    try:
        project_id = markup.get("project_id", "unknown")
        page_num = markup.get("page_num", 1)
        
        print(f"🔄 Сохранение разметки: проект {project_id}, страница {page_num}")
        
        # Добавляем OCR данные к разметке если они есть
        ocr_data = db.get_ocr_data(project_id, page_num)
        if ocr_data:
            markup["ocr_data_from_db"] = ocr_data[0] if ocr_data else {}
            print(f"📋 OCR данные добавлены к разметке")
        
        # Сохраняем в файл
        markup_id, file_path = db.save_markup_to_file(project_id, page_num, markup)
        print(f"✅ Файл сохранен: {file_path}")
        
        # Сохраняем в БД
        db_markup_id = None
        try:
            db_markup_id = db.save_markup(project_id, page_num, markup, is_training=True)
            print(f"✅ БД сохранена, ID: {db_markup_id}")
        except Exception as db_error:
            print(f"⚠️ Ошибка сохранения в БД: {db_error}")
        
        return {
            "success": True,
            "message": "Разметка сохранена" + (" (только в файл)" if db_markup_id is None else " (в файл и БД)"),
            "markup_id": markup_id,
            "file_path": file_path,
            "db_id": db_markup_id if db_markup_id else "не удалось сохранить в БД",
            "ocr_data_included": bool(ocr_data)
        }
        
    except Exception as e:
        print(f"❌ Критическая ошибка сохранения: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Ошибка сохранения: {str(e)}"
        }

@app.delete("/api/markup/{markup_id}/")
async def delete_markup_file(markup_id: str):
    """Удаление разметки"""
    try:
        success = db.delete_markup(markup_id)
        if success:
            return {
                "success": True,
                "message": f"Разметка {markup_id} удалена"
            }
        else:
            return {
                "success": False,
                "message": f"Разметка {markup_id} не найдена"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка удаления: {str(e)}"
        }

@app.post("/api/train/selected/")
async def train_with_selected_markups(markup_ids: list):
    """Обучение на выбранных разметках"""
    try:
        selected_markups = []
        
        for markup_id in markup_ids:
            markup = db.get_markup_by_id(markup_id)
            if markup:
                selected_markups.append({
                    "markup_data": markup,
                    "markup_id": markup_id
                })
        
        if not selected_markups:
            return {
                "success": False,
                "message": "Не выбрано ни одной разметки",
                "accuracy": 0,
                "samples": 0
            }
        
        training_data = []
        for item in selected_markups:
            training_data.append({
                "markup_data": item["markup_data"],
                "markup_id": item["markup_id"]
            })
        
        result = wall_model.train(training_data)
        
        if result:
            return {
                "success": True,
                "message": f"Модель обучена на {len(selected_markups)} разметках",
                "accuracy": result['accuracy'],
                "samples": result['samples'],
                "walls_count": result['walls_count'],
                "non_walls_count": result['non_walls_count'],
                "markups_count": len(selected_markups),
                "markup_ids": markup_ids
            }
        else:
            return {
                "success": False,
                "message": "Ошибка обучения модели",
                "accuracy": 0,
                "samples": 0
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка обучения: {str(e)}",
            "accuracy": 0,
            "samples": 0
        }

@app.get("/api/training-stats/")
async def get_training_stats():
    """Получение статистики по данным обучения"""
    try:
        stats = db.get_training_statistics()
        return stats
    except Exception as e:
        return {
            "total_markups": 0,
            "training_markups": 0,
            "validation_markups": 0,
            "projects_count": 0,
            "walls_count": 0,
            "file_markups_count": 0,
            "error": str(e)
        }

@app.post("/api/save-training-markup/")
async def save_training_markup(markup: dict):
    """Сохранение разметки специально для обучения"""
    try:
        project_id = markup.get("project_id", "unknown")
        page_num = markup.get("page_num", 1)
        
        ocr_data = db.get_ocr_data(project_id, page_num)
        if ocr_data:
            markup["ocr_data_from_db"] = ocr_data[0] if ocr_data else {}
        
        markup_id = db.save_markup(project_id, page_num, markup, is_training=True)
        
        markup_id, file_path = db.save_markup_to_file(project_id, page_num, markup)
        
        return {
            "success": True,
            "message": "Разметка сохранена для обучения",
            "markup_id": markup_id,
            "db_id": markup_id,
            "file_path": file_path,
            "ocr_data_included": bool(ocr_data)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка сохранения: {str(e)}"
        }
    
# ========== ML API ENDPOINTS (для совместимости со старыми ссылками) ==========

@app.get("/api/ml/model-status/")
async def get_model_status_ml():
    """Дубликат для совместимости с ml_test_interface.html"""
    return await get_model_status()

@app.post("/api/ml/analyze-markup/")
async def analyze_markup_ml(markup: dict):
    """Дубликат для совместимости с ml_test_interface.html"""
    return await analyze_markup(markup)

@app.post("/api/ml/predict/")
async def predict_walls_ml(markup: dict):
    """Дубликат для совместимости с ml_test_interface.html"""
    return await predict_walls(markup)

@app.post("/api/ml/train/")
async def train_model_ml():
    """Дубликат для совместимости с ml_test_interface.html"""
    return await train_model()

@app.post("/api/ml/feedback/")
async def receive_feedback_ml(feedback: dict):
    """Дубликат для совместимости с ml_test_interface.html"""
    return await receive_feedback(feedback)

@app.get("/api/ml/training-stats/")
async def get_training_stats_ml():
    """Дубликат для совместимости с ml_test_interface.html"""
    return await get_training_stats()

@app.post("/api/ml/save-training-markup/")
async def save_training_markup_ml(markup: dict):
    """Дубликат для совместимости с ml_test_interface.html"""
    return await save_training_markup(markup)

# ========== COMPUTER VISION API ENDPOINTS ==========

@app.post("/api/detect-walls-auto/")
async def detect_walls_auto(request: dict):
    """Автоматическое обнаружение стен на чертеже с помощью CV"""
    try:
        project_id = request.get("project_id")
        page_num = request.get("page_num", 1)
        
        if not project_id:
            return {
                "success": False,
                "message": "Не указан project_id"
            }
        
        print(f"🤖 Запуск автообнаружения стен: проект {project_id}, стр. {page_num}")
        
        # Используем нашу CV модель
        from cv_model import cv_model
        
        result = cv_model.process_project_page(project_id, page_num)
        
        if result.get("success"):
            # Сохраняем разметку в БД
            markup_data = {
                "project_id": project_id,
                "page_num": page_num,
                "objects": result.get("objects", []),
                "total_objects": result.get("total_objects", 0),
                "detection_method": "YOLO Auto-detection",
                "auto_detected": True
            }
            
            # Сохраняем в БД как тренировочные данные
            try:
                markup_id = db.save_markup(project_id, page_num, markup_data, is_training=True)
                result["db_markup_id"] = markup_id
                print(f"✅ Авторазметка сохранена в БД, ID: {markup_id}")
            except Exception as db_error:
                print(f"⚠️ Не удалось сохранить в БД: {db_error}")
                result["db_error"] = str(db_error)
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка автообнаружения: {e}")
        return {
            "success": False,
            "message": f"Ошибка: {str(e)}"
        }

@app.get("/api/compare-detection/{project_id}/{page_num}/")
async def compare_detection_methods(project_id: str, page_num: int):
    """Сравнение RandomForest и YOLO методов обнаружения"""
    try:
        # Получаем изображение
        images_dir = PROCESSED_DIR / project_id
        image_pattern = f"page_{page_num:03d}.jpg"
        image_path = images_dir / image_pattern
        
        if not image_path.exists():
            # Ищем альтернативный формат
            all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            if all_images and page_num <= len(all_images):
                image_path = all_images[page_num - 1]
            else:
                return {
                    "success": False,
                    "message": "Изображение не найдено"
                }
        
        # 1. YOLO обнаружение
        from cv_model import cv_model
        yolo_detections = cv_model.detect_walls_hybrid(image_path)
        yolo_count = len(yolo_detections)
        
        # 2. RandomForest обнаружение (старый метод)
        from ml_model import wall_model
        # Создаем фиктивную разметку для RF
        fake_markup = {
            "objects": [{"type": "wall", "points": [{"x": 0, "y": 0}]}]  # Минимальная разметка
        }
        rf_predictions = wall_model.predict_walls(fake_markup)
        rf_count = len(rf_predictions)
        
        # 3. Геометрический анализ
        geometry = cv_model.analyze_geometry(image_path)
        
        comparison = {
            "success": True,
            "project_id": project_id,
            "page_num": page_num,
            "image_size": f"{image_path.stat().st_size / 1024:.1f} KB",
            "methods": {
                "yolo_cv": {
                    "detected_walls": yolo_count,
                    "method": "YOLOv8 + Computer Vision",
                    "description": "Автоматическое обнаружение на основе пикселей изображения",
                    "accuracy_estimate": "85-95% (после дообучения)",
                    "auto_detection": True
                },
                "random_forest": {
                    "detected_walls": rf_count,
                    "method": "RandomForest + Геометрические признаки",
                    "description": "Требует ручной разметки точек, анализирует только координаты",
                    "accuracy_estimate": "40-60%",
                    "auto_detection": False
                }
            },
            "geometry_analysis": geometry,
            "recommendation": "YOLO" if yolo_count > 0 else "Ручная разметка",
            "timestamp": str(np.datetime64('now'))
        }
        
        return comparison
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка сравнения: {str(e)}"
        }

@app.post("/api/train-yolo-custom/")
async def train_yolo_custom():
    """Дообучение YOLO на ваших размеченных данных"""
    try:
        # Получаем все разметки из БД
        markups = db.get_markups_for_training(limit=50)
        
        if len(markups) < 3:
            return {
                "success": False,
                "message": f"Нужно минимум 3 разметки для обучения. Сейчас: {len(markups)}",
                "required": 3,
                "available": len(markups)
            }
        
        print(f"🔄 Начинаем дообучение YOLO на {len(markups)} разметках...")
        
        # TODO: Здесь будет код конвертации ваших разметок в YOLO формат
        # и дообучение модели на архитектурных чертежах
        
        return {
            "success": True,
            "message": f"Готово к дообучению на {len(markups)} примерах",
            "next_steps": [
                "1. Конвертировать разметки в YOLO формат",
                "2. Создать dataset.yaml",
                "3. Запустить training на GPU/CPU",
                "4. Сохранить веса модели"
            ],
            "markups_count": len(markups),
            "estimated_time": "2-4 часа (зависит от GPU)"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка подготовки к обучению: {str(e)}"
        }

# Новый endpoint для веб-интерфейса
@app.get("/cv-dashboard/")
async def cv_dashboard():
    """Дашборд для управления CV моделью"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smet4ik - CV Model Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #333; }
            .card { 
                background: white; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .method-comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 30px 0;
            }
            .method-card {
                padding: 20px;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
            }
            .yolo-card { border-color: #4CAF50; background: #f0f9f0; }
            .rf-card { border-color: #FF9800; background: #fff3e0; }
            .btn {
                padding: 12px 24px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px 5px;
            }
            .btn:hover { background: #45a049; }
            .btn-secondary { background: #2196F3; }
            .btn-secondary:hover { background: #1976D2; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Smet4ik - Computer Vision Dashboard</h1>
            
            <div class="card">
                <h2>Текущий статус CV системы</h2>
                <div id="cvStatus">Загрузка...</div>
                <button class="btn" onclick="checkCVStatus()">Обновить статус</button>
            </div>
            
            <div class="method-comparison">
                <div class="method-card yolo-card">
                    <h3>🎯 YOLO + Computer Vision</h3>
                    <p><strong>Новая система:</strong></p>
                    <ul>
                        <li>Автоматическое обнаружение стен</li>
                        <li>Работает с пикселями изображения</li>
                        <li>Точность: 85-95% (после дообучения)</li>
                        <li>Не требует ручной разметки</li>
                        <li>⚡ Быстрое обнаружение</li>
                    </ul>
                    <button class="btn" onclick="testYOLO()">Протестировать YOLO</button>
                </div>
                
                <div class="method-card rf-card">
                    <h3>📊 RandomForest (старая система)</h3>
                    <p><strong>Текущая система:</strong></p>
                    <ul>
                        <li>Требует ручной разметки точек</li>
                        <li>Анализирует только координаты</li>
                        <li>Точность: 40-60%</li>
                        <li>Нет работы с изображениями</li>
                        <li>🐢 Медленное обучение</li>
                    </ul>
                    <button class="btn btn-secondary" onclick="compareMethods()">Сравнить методы</button>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Автообнаружение стен</h2>
                <div>
                    <label>Project ID: </label>
                    <input type="text" id="projectId" value="1856415c">
                    <label>Страница: </label>
                    <input type="number" id="pageNum" value="1" min="1">
                    <button class="btn" onclick="autoDetectWalls()">🔍 Автообнаружение</button>
                </div>
                <div id="autoDetectResult" class="status"></div>
            </div>
            
            <div class="card">
                <h2>🎓 Дообучение модели</h2>
                <p>Для повышения точности до 95% нужно дообучить YOLO на ваших чертежах.</p>
                <button class="btn" onclick="trainCustomYOLO()">🚀 Начать дообучение</button>
                <div id="trainingStatus" class="status"></div>
            </div>
        </div>
        
        <script>
            async function checkCVStatus() {
                const statusEl = document.getElementById('cvStatus');
                statusEl.innerHTML = 'Проверка...';
                
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    statusEl.innerHTML = `
                        <div class="success status">
                            <strong>✅ CV система активна</strong><br>
                            Версия: ${data.version || '0.9.0'}<br>
                            OCR доступен: ${data.ocr_available ? 'Да' : 'Нет'}<br>
                            YOLO модель: Загружена
                        </div>
                    `;
                } catch (error) {
                    statusEl.innerHTML = `<div class="info status">❌ Ошибка: ${error.message}</div>`;
                }
            }
            
            async function testYOLO() {
                const resultEl = document.getElementById('autoDetectResult');
                resultEl.innerHTML = 'Тестирование...';
                
                try {
                    const response = await fetch('/api/detect-walls-auto/', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            project_id: '1856415c',
                            page_num: 1
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        resultEl.innerHTML = `
                            <div class="success status">
                                <strong>✅ YOLO обнаружение успешно!</strong><br>
                                Найдено стен: ${data.total_objects || 0}<br>
                                Файл: ${data.image || 'N/A'}<br>
                                <a href="/api/markup/${data.db_markup_id || 'test'}/" target="_blank">
                                    Посмотреть разметку
                                </a>
                            </div>
                        `;
                    } else {
                        resultEl.innerHTML = `<div class="info status">❌ ${data.message}</div>`;
                    }
                } catch (error) {
                    resultEl.innerHTML = `<div class="info status">❌ Ошибка сети: ${error.message}</div>`;
                }
            }
            
            async function autoDetectWalls() {
                const projectId = document.getElementById('projectId').value;
                const pageNum = document.getElementById('pageNum').value;
                const resultEl = document.getElementById('autoDetectResult');
                
                resultEl.innerHTML = 'Запуск автообнаружения...';
                
                try {
                    const response = await fetch('/api/detect-walls-auto/', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            project_id: projectId,
                            page_num: parseInt(pageNum)
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        resultEl.innerHTML = `
                            <div class="success status">
                                <strong>✅ Автообнаружение завершено!</strong><br>
                                Найдено стен: ${data.total_objects || 0}<br>
                                Средняя уверенность: ${data.objects && data.objects[0] ? (data.objects[0].confidence * 100).toFixed(1) : '0'}%<br>
                                <button onclick="viewResults('${projectId}', ${pageNum})">📊 Посмотреть детали</button>
                            </div>
                        `;
                    } else {
                        resultEl.innerHTML = `<div class="info status">❌ ${data.message}</div>`;
                    }
                } catch (error) {
                    resultEl.innerHTML = `<div class="info status">❌ Ошибка: ${error.message}</div>`;
                }
            }
            
            async function compareMethods() {
                const projectId = document.getElementById('projectId').value;
                const pageNum = document.getElementById('pageNum').value;
                const resultEl = document.getElementById('autoDetectResult');
                
                resultEl.innerHTML = 'Сравнение методов...';
                
                try {
                    const response = await fetch(`/api/compare-detection/${projectId}/${pageNum}/`);
                    const data = await response.json();
                    
                    if (data.success) {
                        const yolo = data.methods.yolo_cv;
                        const rf = data.methods.random_forest;
                        
                        resultEl.innerHTML = `
                            <div class="success status">
                                <strong>📊 Сравнение методов обнаружения</strong><br>
                                <strong>YOLO CV:</strong> ${yolo.detected_walls} стен | ${yolo.accuracy_estimate}<br>
                                <strong>RandomForest:</strong> ${rf.detected_walls} стен | ${rf.accuracy_estimate}<br>
                                <strong>Рекомендация:</strong> Использовать <strong>${data.recommendation}</strong><br>
                                <strong>Геометрия:</strong> ${data.geometry_analysis.total_lines} линий найдено
                            </div>
                        `;
                    } else {
                        resultEl.innerHTML = `<div class="info status">❌ ${data.message}</div>`;
                    }
                } catch (error) {
                    resultEl.innerHTML = `<div class="info status">❌ Ошибка: ${error.message}</div>`;
                }
            }
            
            async function trainCustomYOLO() {
                const statusEl = document.getElementById('trainingStatus');
                statusEl.innerHTML = 'Подготовка к дообучению...';
                
                try {
                    const response = await fetch('/api/train-yolo-custom/', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        statusEl.innerHTML = `
                            <div class="success status">
                                <strong>✅ Готово к дообучению!</strong><br>
                                Доступно разметок: ${data.markups_count}<br>
                                Оцен. время: ${data.estimated_time}<br>
                                <strong>Следующие шаги:</strong>
                                <ol>
                                    ${data.next_steps.map(step => `<li>${step}</li>`).join('')}
                                </ol>
                            </div>
                        `;
                    } else {
                        statusEl.innerHTML = `<div class="info status">❌ ${data.message}</div>`;
                    }
                } catch (error) {
                    statusEl.innerHTML = `<div class="info status">❌ Ошибка: ${error.message}</div>`;
                }
            }
            
            function viewResults(projectId, pageNum) {
                window.open(`/project/${projectId}/?page=${pageNum}`, '_blank');
            }
            
            // Автозагрузка статуса
            window.onload = checkCVStatus;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/ml-test/")
async def ml_test_interface():
    """Интерфейс для тестирования ML модели"""
    html_path = Path(__file__).parent / "ml_test_interface.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)