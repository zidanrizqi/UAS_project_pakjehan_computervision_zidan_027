# =========================
# IMPORT LIBRARY
# =========================

import cv2                          # OpenCV untuk computer vision
import os                           # Untuk manajemen file & direktori
import json                         # Untuk membaca file label wajah (JSON)
import csv                          # Untuk menyimpan data absensi (CSV)
from datetime import datetime       # Untuk mencatat waktu absensi


# =========================
# BASE DIRECTORY
# =========================

# Mengambil direktori tempat file Python ini berada
# Ini penting agar file CSV, model, dan asset selalu terdeteksi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# LOAD MODEL FACE RECOGNITION
# =========================

# Membuat objek LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load model hasil training
recognizer.read(
    os.path.join(BASE_DIR, "assets", "face_model.xml")
)


# =========================
# LOAD LABEL ID → NAMA
# =========================

# Membaca mapping label (id) ke nama orang
with open(
    os.path.join(BASE_DIR, "assets", "labels.json"),
    "r",
    encoding="utf-8"
) as f:
    label_ids = json.load(f)


# =========================
# LOAD HAAR CASCADE
# =========================

# Path file Haar Cascade untuk deteksi wajah
cascade_path = os.path.join(
    BASE_DIR, "assets", "haarcascade_frontalface_default.xml"
)

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)


# =========================
# SETUP FILE ABSENSI CSV
# =========================

# Menentukan path file absensi
ABSEN_FILE = os.path.join(BASE_DIR, "absen.csv")

# Jika file absensi belum ada, buat file baru + header
if not os.path.exists(ABSEN_FILE):
    with open(ABSEN_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nama", "waktu"])


# =========================
# PENANDA AGAR ABSEN 1X SAJA
# =========================

# Set untuk menyimpan nama yang sudah absen
already_absent = set()


# =========================
# FUNGSI ABSENSI
# =========================

def attendance(name):
    """
    Mencatat absensi ke file CSV
    Setiap nama hanya dicatat satu kali
    """

    # Jika nama sudah pernah absen, hentikan fungsi
    if name in already_absent:
        return

    # Tandai nama sudah absen
    already_absent.add(name)

    # Ambil waktu sekarang
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tampilkan ke terminal
    print(f"[ABSEN] {name} - {waktu}")

    # Simpan ke file CSV
    with open(ABSEN_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, waktu])


# =========================
# AKSES WEBCAM
# =========================

# Membuka kamera default (0)
cap = cv2.VideoCapture(0)


# =========================
# LOOP UTAMA PROGRAM
# =========================

while True:
    # Mengambil satu frame dari webcam
    ret, frame = cap.read()

    # Jika kamera gagal membaca frame, hentikan loop
    if not ret:
        break

    # Mengubah frame ke grayscale (wajib untuk Haar & LBPH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah pada frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Loop setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Mengambil area wajah (Region of Interest)
        face_roi = gray[y:y+h, x:x+w]

        # Prediksi identitas wajah
        label, confidence = recognizer.predict(face_roi)

        # Mengambil nama dari label
        name = label_ids.get(str(label), "Unknown")

        # Membuat teks untuk ditampilkan
        text = f"{name} ({confidence:.1f})"

        # =========================
        # LOGIKA ABSENSI
        # =========================

        # Semakin kecil confidence → semakin mirip
        # Hanya absen jika confidence cukup bagus
        if confidence < 40 and name != "Unknown":
            attendance(name)

        # =========================
        # TAMPILKAN HASIL KE LAYAR
        # =========================

        # Menampilkan nama & confidence
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Menggambar kotak di sekitar wajah
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Menampilkan window kamera
    cv2.imshow("Face Recognition + Absensi", frame)

    # Tekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# BERSIHKAN RESOURCE
# =========================

cap.release()             # Melepaskan kamera
cv2.destroyAllWindows()   # Menutup semua window OpenCV
