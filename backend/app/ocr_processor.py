# ocr_processor.py
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from pathlib import Path
import os
import json

class OCRProcessor:
    def __init__(self, tesseract_path=None):
        """
        Инициализация OCR процессора для чертежей
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Автоматический поиск в стандартных путях
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Tesseract найден: {path}")
                    break
            else:
                print("⚠️ Tesseract не найден в стандартных путях")
    
    def extract_text_from_image(self, image_path):
        """
        Извлечение текста из изображения чертежа
        """
        try:
            # Загружаем изображение
            image = cv2.imread(str(image_path))
            if image is None:
                return ""
            
            # Конвертируем в grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Увеличиваем контраст
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Бинаризация
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Применяем OCR (русский + английский)
            text = pytesseract.image_to_string(
                binary, 
                config='--oem 3 --psm 6',
                lang='rus+eng'
            )
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ Ошибка OCR: {e}")
            return ""
    
    def extract_measurements(self, image_path):
        """
        Извлечение измерений из чертежа
        Возвращает: список найденных размеров в миллиметрах
        """
        text = self.extract_text_from_image(image_path)
        
        if not text:
            return []
        
        # Паттерны для поиска размеров (в миллиметрах)
        patterns = [
            # 3500 мм, 1200 мм
            (r'(\d+(?:[.,]\d+)?)\s*(?:мм|mm|м|m)', 1),
            # 1200x1500 мм, 1200 x 1500 мм
            (r'(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm|м|m)', 2),
            # R100 мм, Ø100 мм
            (r'(?:R|Ø)\s*(\d+(?:[.,]\d+)?)\s*(?:мм|mm|м|m)', 1),
        ]
        
        measurements = []
        
        for pattern, group_count in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if group_count == 1:
                    value = float(match.group(1).replace(',', '.'))
                    measurements.append({
                        'value_mm': value,
                        'text': match.group(0),
                        'type': 'linear'
                    })
                elif group_count == 2:
                    value1 = float(match.group(1).replace(',', '.'))
                    value2 = float(match.group(2).replace(',', '.'))
                    measurements.append({
                        'value_mm': value1,
                        'value2_mm': value2,
                        'text': match.group(0),
                        'type': 'rectangular'
                    })
        
        return measurements
    
    def analyze_page(self, image_path):
        """
        Полный анализ страницы чертежа
        """
        print(f"🔍 Анализ страницы: {image_path}")
        
        # Извлекаем текст
        text = self.extract_text_from_image(image_path)
        
        # Извлекаем измерения
        measurements = self.extract_measurements(image_path)
        
        # Анализ текста на наличие ключевых слов
        keywords = {
            'стена': ['стен', 'стена', 'стены', 'wall'],
            'окно': ['окн', 'окно', 'окна', 'window'],
            'дверь': ['двер', 'дверь', 'двери', 'door'],
            'комната': ['комнат', 'комната', 'room'],
            'кухня': ['кухн', 'кухня', 'kitchen'],
            'ванная': ['ванн', 'ванная', 'bathroom']
        }
        
        found_keywords = []
        text_lower = text.lower()
        
        for category, words in keywords.items():
            for word in words:
                if word in text_lower:
                    found_keywords.append(category)
                    break
        
        result = {
            'page_path': str(image_path),
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'total_text_length': len(text),
            'measurements_count': len(measurements),
            'measurements': measurements,
            'keywords': list(set(found_keywords)),
            'has_architectural_data': len(measurements) > 0 or len(found_keywords) > 0
        }
        
        return result

# Глобальный экземпляр
ocr_processor = OCRProcessor()