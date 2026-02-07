# run.py - Точка входа для запуска сервера
import uvicorn
import sys
import os

# Добавляем текущую директорию в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Запуск Smet4ik AI Trainer...")
    print("📁 Текущая директория:", os.getcwd())
    print("🔌 API будет доступно по адресу: http://127.0.0.1:8000")
    print("📖 Документация API: http://127.0.0.1:8000/docs")
    print("🎨 Интерфейс разметки: http://127.0.0.1:8000/marker/")
    print("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )