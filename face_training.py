import cv2                  # Library OpenCV untuk pengolahan citra dan face recognition
import os                   # Library untuk operasi file dan folder
import numpy as np          # Library NumPy untuk pengolahan array numerik


def load_dataset(dataset_path="dataset"):
    # Fungsi untuk membaca dataset wajah dari folder
    # dataset_path : folder utama dataset (default = "dataset")

    faces = []
    # List untuk menyimpan data gambar wajah (grayscale)

    labels = []
    # List untuk menyimpan label numerik setiap wajah

    label_ids = {}
    # Dictionary untuk mapping label (angka) → nama orang

    current_label = 0
    # Label angka yang akan diberikan ke setiap orang

    for person_name in os.listdir(dataset_path):
        # Loop setiap folder orang di dalam dataset

        person_path = os.path.join(dataset_path, person_name)
        # Membentuk path lengkap folder orang

        # Skip jika bukan folder (misalnya file lain)
        if not os.path.isdir(person_path):
            continue

        label_ids[current_label] = person_name
        # Simpan mapping label → nama orang

        for img_name in os.listdir(person_path):
            # Loop setiap file gambar di folder orang

            img_path = os.path.join(person_path, img_name)
            # Path lengkap ke file gambar

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Membaca gambar dan langsung mengubah ke grayscale
            # LBPH membutuhkan input grayscale

            if img is None:
                # Jika gambar gagal dibaca, lewati
                continue

            faces.append(img)
            # Menambahkan gambar wajah ke list faces

            labels.append(current_label)
            # Menambahkan label sesuai orangnya

        current_label += 1
        # Pindah ke label berikutnya untuk orang selanjutnya

    return faces, labels, label_ids
    # Mengembalikan data wajah, label, dan mapping label → nama


# =========================
# PROSES TRAINING MODEL
# =========================

faces, labels, label_ids = load_dataset("dataset")
# Memanggil fungsi load_dataset untuk mengambil data training

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Membuat objek LBPH Face Recognizer

recognizer.train(faces, np.array(labels))
# Melatih model menggunakan data wajah dan label

recognizer.save("assets/face_model.xml")
# Menyimpan model hasil training ke file XML


# =========================
# SIMPAN LABEL KE FILE JSON
# =========================

import json
# Import JSON untuk menyimpan mapping label ke nama

with open("assets/labels.json", "w") as f:
    # Membuka file labels.json dalam mode tulis
    json.dump(label_ids, f)
    # Menyimpan dictionary label_ids ke file JSON
