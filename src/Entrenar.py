from ultralytics import YOLO

# Ruta al archivo de configuración del dataset
data_yaml = "dataset/data.yaml"

# Modelo base preentrenado (puedes cambiar a yolov8s.pt, yolov8m.pt, etc.)
#model_name = "model/yolov8s.pt"

model_name = "model/yolov10s.pt"

# Crear el modelo a partir del modelo preentrenado
model = YOLO(model_name)

# Entrenar el modelo
#model.train(data=data_yaml, epochs=10, imgsz=640, batch=16, name="yolo_custom")

model.train(data=data_yaml, epochs=100, imgsz=640, batch=8, device="cpu", name="yolo10_custom")