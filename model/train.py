from ultralytics import YOLO

# Cargar modelo base (nano = ligero, ideal para celular)
model = YOLO("yolov8n.pt")  # Descarga automática la primera vez

# Entrenar con tu dataset personalizado
# Tu dataset debe tener la estructura: datasets/football/
model.train(
    data="datasets/football/data.yaml",  # archivo de configuración
    epochs=80,
    imgsz=416,
    batch=16,
    name="yolov8n-football-final",
    patience=15,
    device="0" if torch.cuda.is_available() else "cpu",
    project="runs/football",
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    conf=0.45,
    iou=0.6
)

# Exportar a formato ligero para celular
model.export(format="torchscript")  # o "onnx" si prefieres
print("Modelo entrenado y exportado: best.pt → yolov8n-football.pt")
