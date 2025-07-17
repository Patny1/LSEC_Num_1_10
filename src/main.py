from ultralytics import YOLO
import cv2

# Cargar modelo entrenado
# parámetros model.train(data=data_yaml, epochs=10, imgsz=640, batch=16, name="yolo_custom")
#model = YOLO('runs/detect/yolo_custom3/weights/best.pt')  #

#model.train(data=data_yaml, epochs=100, imgsz=640, batch=8, device="cpu", name="yolo10_custom")
#model = YOLO('runs/detect/yolo10s_CPU/weights/best.pt')

#model GPU epochs=100, imgsz=640, batch=16, device="GPU"
model = YOLO('runs/detect/yolo10s_GPU/weights/best.pt')

#colab
#model=YOLO('runs/COLAB/yolo10_adam_lr005/weights/best.pt')
#model=YOLO('runs/COLAB/yolo10_sgd_lr1/weights/best.pt')
#model=YOLO('runs/COLAB/yolo8s_adam/weights/best.pt')
#model=YOLO('runs/COLAB/yolo8s_batch16/weights/best.pt')
# Iniciar cámara
cap = cv2.VideoCapture(2)  # Usa 0 o cambia por otro índice si tienes múltiples cámaras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia
    results = model(frame, device='cpu')  # Usa CPU
    annotated_frame = results[0].plot()   # Dibuja los resultados

    # Mostrar en pantalla
    cv2.imshow("YOLO - Webcam", annotated_frame)

    # Salir con ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
