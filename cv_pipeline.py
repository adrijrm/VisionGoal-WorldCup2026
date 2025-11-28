import torch
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO

class VisionGoalPipeline:
    def __init__(self, model_path="model/yolov8n-football.pt"):
        # Cargar el modelo directamente desde el archivo local
        self.model = YOLO(model_path)
        self.model.conf = 0.55  # Umbral de confianza
        
        # Inicializar variables necesarias para el seguimiento
        self.prev_gray = None
        self.orb = cv2.ORB_create(nfeatures=500)
    
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
    
    def track_ball(self, frame):
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Detectar características ORB
        kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        
        if des1 is None or des2 is None:
            self.prev_gray = gray
            return None
        
        # Emparejar características
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Estimar movimiento del centro del balón
                ball_center_prev = np.array([[150, 100]], dtype='float32').reshape(-1, 1, 2)
                ball_center_now = cv2.perspectiveTransform(ball_center_prev, M)
                motion = ball_center_now - ball_center_prev
                self.prev_gray = gray
                return motion.flatten()
        
        self.prev_gray = gray
        return None
