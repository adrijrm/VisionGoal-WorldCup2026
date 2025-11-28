import cv2
import numpy as np
from ultralytics import YOLO

class VisionGoalPipeline:
    def __init__(self, model_path="model/yolov8n-football.pt"):
        # Cargar el modelo directamente desde el archivo local
        # Esto no requiere conexión a internet
        self.model = YOLO(model_path)
        self.model.conf = 0.55  # Umbral de confianza

    def preprocess(self, frame):
        # Preprocesamiento: filtrado gaussiano y segmentación del campo
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return cv2.bitwise_and(blurred, blurred, mask=mask)

    def detect(self, frame):
        # Realizar detección usando el modelo cargado localmente
        results = self.model(frame, imgsz=416, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    label = self.model.names[cls]
                    
                    detections.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": label,
                        "conf": float(conf)
                    })
        return detections
