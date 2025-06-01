import torch
print(torch.__version__)

import time
import cv2
from ultralytics import YOLO, RTDETR

# Load model YOLOv8 atau RT-DETR
# model = YOLO("Models/YOLOv8m/best.pt")  # Jika pakai YOLOv8
model = RTDETR("Models/RT-DETR/best.pt")  # Jika pakai RT-DETR

# Buka kamera (0 = kamera default, 1 = eksternal)
cap = cv2.VideoCapture(1)

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("❌ Tidak bisa membuka kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame.")
        break

    # Mulai hitung waktu untuk FPS
    start_time = time.time()

    # Inference langsung pakai BGR (tidak perlu konversi RGB)
    results = model(frame, verbose=False)

    # Ambil frame dengan bounding box yang sudah digambar
    annotated_frame = results[0].plot()

    # Hitung FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)

    # Tampilkan FPS di frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame di jendela
    cv2.imshow("RT-DETR Real-time Detection", annotated_frame)

    # Tekan ESC (kode 27) untuk keluar
    if cv2.waitKey(1) == 27:
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
