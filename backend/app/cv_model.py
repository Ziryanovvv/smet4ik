# cv_model.py - УЛУЧШЕННАЯ ВЕРСИЯ
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
import torch
import math

class WallDetectionCVModel:
    """
    Улучшенная модель компьютерного зрения для обнаружения стен на чертежах
    Комбинирует YOLOv8 и геометрический анализ
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        
        print(f"🔧 Инициализация улучшенной CV модели...")
        print(f"   Устройство: {self.device}")
        
        self.load_model()
    
    def load_model(self):
        """Загрузка модели YOLO"""
        try:
            # Используем YOLOv8-seg для сегментации (лучше для стен)
            self.model = YOLO('yolov8n-seg.pt')  # Сегментационная модель
            print("✅ Загружена предобученная модель YOLOv8n-seg")
            
            self.model_loaded = True
            print(f"   Модель готова к работе на {self.device}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model_loaded = False
    
    def analyze_geometry(self, image_path: Path) -> Dict[str, Any]:
        """
        Геометрический анализ чертежа для поиска стен
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Геометрические признаки
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {}
            
            # Конвертируем в grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Применяем детектор границ Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Находим линии с помощью преобразования Хафа
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=50, 
                minLineLength=100, 
                maxLineGap=10
            )
            
            geometric_features = {
                'total_lines': 0,
                'horizontal_lines': 0,
                'vertical_lines': 0,
                'diagonal_lines': 0,
                'avg_line_length': 0,
                'line_detected': False
            }
            
            if lines is not None:
                geometric_features['total_lines'] = len(lines)
                lengths = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    lengths.append(length)
                    
                    # Определяем ориентацию линии
                    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                    
                    if angle < 10 or angle > 170:  # Горизонтальная
                        geometric_features['horizontal_lines'] += 1
                    elif 80 < angle < 100:  # Вертикальная
                        geometric_features['vertical_lines'] += 1
                    else:  # Диагональная
                        geometric_features['diagonal_lines'] += 1
                
                if lengths:
                    geometric_features['avg_line_length'] = sum(lengths) / len(lengths)
                    geometric_features['line_detected'] = True
            
            return geometric_features
            
        except Exception as e:
            print(f"⚠️ Ошибка геометрического анализа: {e}")
            return {}
    
    def detect_walls_hybrid(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Гибридное обнаружение стен: YOLO + Геометрический анализ
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список обнаруженных стен
        """
        if not self.model_loaded:
            return []
        
        try:
            print(f"🔍 Гибридный анализ: {image_path.name}")
            
            # 1. YOLO обнаружение
            results = self.model(
                source=str(image_path),
                conf=0.2,  # Более низкий порог для чертежей
                device=self.device,
                verbose=False
            )
            
            # 2. Геометрический анализ
            geometry = self.analyze_geometry(image_path)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls]
                        
                        # Фильтруем по геометрическим признакам
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Признаки для определения стены:
                        # 1. Соотношение сторон (стены обычно длинные и узкие)
                        # 2. Наличие горизонтальных/вертикальных линий
                        # 3. Размер относительно изображения
                        
                        is_wall_like = False
                        wall_confidence = float(conf)
                        
                        if geometry['line_detected']:
                            # Увеличиваем уверенность если есть линии
                            wall_confidence *= 1.2
                        
                        # Проверяем признаки стены
                        if (0.5 < aspect_ratio < 20 or  # Длинная форма
                            width > 100 or height > 100):  # Достаточно большой размер
                            is_wall_like = True
                        
                        if is_wall_like or wall_confidence > 0.3:
                            detection = {
                                'type': 'wall',
                                'confidence': min(wall_confidence, 1.0),
                                'bbox': {
                                    'x1': float(x1),
                                    'y1': float(y1),
                                    'x2': float(x2),
                                    'y2': float(y2)
                                },
                                'dimensions': {
                                    'width_px': float(width),
                                    'height_px': float(height),
                                    'aspect_ratio': float(aspect_ratio)
                                },
                                'geometry_info': geometry,
                                'center': {
                                    'x': float((x1 + x2) / 2),
                                    'y': float((y1 + y2) / 2)
                                }
                            }
                            detections.append(detection)
            
            print(f"✅ Найдено возможных стен: {len(detections)}")
            if geometry['line_detected']:
                print(f"📏 Геометрия: {geometry['horizontal_lines']} гориз., {geometry['vertical_lines']} верт. линий")
            
            return detections
            
        except Exception as e:
            print(f"❌ Ошибка гибридного обнаружения: {e}")
            return []
    
    def convert_to_markup_format(self, detections: List[Dict], 
                                image_path: Path) -> Dict[str, Any]:
        """
        Конвертация обнаружений в формат разметки
        
        Args:
            detections: Список обнаружений
            image_path: Путь к изображению
            
        Returns:
            Данные в формате разметки
        """
        markup_objects = []
        
        for det in detections:
            bbox = det['bbox']
            
            # Преобразуем bounding box в полигон (4 точки)
            points = [
                {'x': bbox['x1'], 'y': bbox['y1']},
                {'x': bbox['x2'], 'y': bbox['y1']},
                {'x': bbox['x2'], 'y': bbox['y2']},
                {'x': bbox['x1'], 'y': bbox['y2']}
            ]
            
            obj = {
                'type': 'wall',
                'points': points,
                'confidence': det['confidence'],
                'dimensions': det['dimensions'],
                'center': det['center']
            }
            
            markup_objects.append(obj)
        
        # Получаем размеры изображения
        try:
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
        except:
            width, height = 1000, 1000  # Значения по умолчанию
        
        markup = {
            'project_id': 'auto_detected',
            'page_num': 1,
            'image_dimensions': {
                'width_px': width,
                'height_px': height
            },
            'objects': markup_objects,
            'total_objects': len(markup_objects),
            'detection_method': 'YOLO+Geometry Hybrid',
            'created_at': str(np.datetime64('now')),
            'model_version': 'v1.0-hybrid'
        }
        
        return markup
    
    def process_project_page(self, project_id: str, page_num: int) -> Dict[str, Any]:
        """
        Обработка страницы проекта
        
        Args:
            project_id: ID проекта
            page_num: Номер страницы
            
        Returns:
            Результаты обнаружения
        """
        try:
            print(f"🔍 Поиск изображения для проекта {project_id}, страница {page_num}")
            
            # Пробуем разные пути для поиска изображения
            base_path = Path(__file__).parent.parent  # C:\smet4ik\backend
            
            # 1. Проверяем в processed_images
            processed_path = base_path / "processed_images" / project_id
            print(f"   Путь processed_images: {processed_path}")
            
            # 2. Проверяем в app/processed_images
            app_processed_path = base_path / "app" / "processed_images" / project_id
            print(f"   Путь app/processed_images: {app_processed_path}")
            
            image_path = None
            
            # Сначала ищем в processed_images
            if processed_path.exists():
                patterns = [
                    f"page_{page_num:03d}.jpg",
                    f"page_{page_num}.jpg",
                    f"page_{page_num:03d}.png",
                    f"page_{page_num}.png",
                    f"page_{page_num:03d}.jpeg",
                    f"page_{page_num}.jpeg"
                ]
                
                for pattern in patterns:
                    test_path = processed_path / pattern
                    if test_path.exists():
                        image_path = test_path
                        print(f"✅ Найдено изображение: {image_path}")
                        break
            
            # Если не нашли, ищем в app/processed_images
            if not image_path and app_processed_path.exists():
                patterns = [
                    f"page_{page_num:03d}.jpg",
                    f"page_{page_num}.jpg",
                    f"page_{page_num:03d}.png",
                    f"page_{page_num}.png"
                ]
                
                for pattern in patterns:
                    test_path = app_processed_path / pattern
                    if test_path.exists():
                        image_path = test_path
                        print(f"✅ Найдено изображение в app/: {image_path}")
                        break
            
            # Если все еще не нашли, ищем любой файл изображения
            if not image_path and processed_path.exists():
                all_images = list(processed_path.glob("*.jpg")) + \
                            list(processed_path.glob("*.png")) + \
                            list(processed_path.glob("*.jpeg"))
                
                if all_images and page_num <= len(all_images):
                    all_images.sort()
                    image_path = all_images[page_num - 1]
                    print(f"✅ Используем изображение по номеру: {image_path}")
            
            if not image_path:
                error_msg = f"Изображение не найдено для проекта {project_id}, стр. {page_num}"
                print(f"❌ {error_msg}")
                print(f"   Проверенные пути:")
                print(f"   - {processed_path}")
                print(f"   - {app_processed_path}")
                return {'error': error_msg, 'success': False}
            
            # Выполняем обнаружение
            print(f"🔍 Запуск обнаружения стен на: {image_path.name}")
            detections = self.detect_walls_hybrid(image_path)
            
            if not detections:
                return {
                    'success': False,
                    'message': 'Стены не обнаружены',
                    'image': image_path.name,
                    'project_id': project_id,
                    'page_num': page_num
                }
            
            # Конвертируем в формат разметки
            markup = self.convert_to_markup_format(detections, image_path)
            markup['project_id'] = project_id
            markup['page_num'] = page_num
            markup['success'] = True
            markup['image_path'] = str(image_path)
            
            # Сохраняем в файл
            output_file = base_path / f"auto_detected_{project_id}_p{page_num}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(markup, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Авторазметка сохранена: {output_file}")
            
            return markup
            
        except Exception as e:
            print(f"❌ Ошибка автообнаружения: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'success': False}

# Глобальный экземпляр модели
cv_model = WallDetectionCVModel()