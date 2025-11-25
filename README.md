# VisionGoal – Accesibilidad en el Mundial de Fútbol 2026  
**LIS4042 Artificial Vision – Fall 2025**  
Prof. Zobeida Guzmán  

### Problema identificado (World Cup 2026)
Fans con discapacidad visual no pueden seguir las jugadas en tiempo real dentro del estadio.

### Solución propuesta
App móvil que usa la cámara del celular para:
- Detectar balón, jugadores (equipo rojo/azul), árbitro y porterías
- Seguir el movimiento del balón con optical flow
- Generar descripción de audio en español en tiempo real

### Técnicas implementadas (clásicas + modernas – SIN APIs)
- Gaussian blur + filtrado HSV (classic)
- Sobel edges → Canny-style (classic)
- ORB keypoints (classic)
- YOLOv8-nano entrenado por mí (modern – PyTorch)
- Lucas-Kanade optical flow (classic)
- Síntesis de voz offline (pyttsx3)

### Cómo ejecutar
```bash
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
python main.py                  # abre la webcam y habla
