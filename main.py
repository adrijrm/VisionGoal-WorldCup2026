import os
# Desactivar las verificaciones en línea de Ultralytics para evitar conexiones a internet
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_HUB_NO_VERIFY'] = 'True'

import cv2
from cv_pipeline import VisionGoalPipeline
from audio_feedback import speak

def main():
    # Crear instancia del pipeline
    pipeline = VisionGoalPipeline()
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    print("Iniciando detección. Presione 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame")
            break
        
        # Realizar detección
        detections = pipeline.detect(frame)
        
        # Dibujar detecciones
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label = d["label"]
            conf = d["conf"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("VisionGoal", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
