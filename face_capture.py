import cv2              # Mengimpor library OpenCV untuk kamera dan pengolahan citra
import os               # Mengimpor library OS untuk manajemen folder dan path


def capture_faces(person_name, output_dir="dataset", max_images=50):
    # Fungsi untuk mengambil gambar wajah dari webcam
    # person_name : nama orang (dipakai sebagai nama folder)
    # output_dir  : folder utama dataset
    # max_images  : jumlah maksimal gambar yang diambil

    save_path = os.path.join(output_dir, person_name)
    # Menggabungkan path output_dir dan person_name
    # Contoh: dataset/zidan

    os.makedirs(save_path, exist_ok=True)
    # Membuat folder jika belum ada
    # exist_ok=True → tidak error jika folder sudah ada

    cap = cv2.VideoCapture(0)
    # Membuka webcam default (0)

    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    # Memuat model Haar Cascade untuk deteksi wajah

    count = 0
    # Variabel untuk menghitung jumlah gambar yang disimpan

    print("[INFO] Mulai capture. Tekan Q untuk stop.")
    # Menampilkan pesan informasi ke terminal

    while True:
        # Loop utama untuk membaca frame kamera

        ret, frame = cap.read()
        # ret   : status berhasil atau tidak
        # frame : gambar dari webcam

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Mengubah frame dari BGR ke grayscale
        # Haar Cascade bekerja lebih baik pada grayscale

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Mendeteksi wajah pada gambar grayscale
        # 1.3 = scaleFactor
        # 5   = minNeighbors (akurasi deteksi)

        for x, y, w, h in faces:
            # Loop setiap wajah yang terdeteksi
            # x, y = posisi wajah
            # w, h = lebar dan tinggi wajah

            face_roi = gray[y:y+h, x:x+w]
            # Mengambil area wajah saja (Region of Interest)

            img_path = os.path.join(save_path, f"{count}.png")
            # Menentukan path file gambar
            # Contoh: dataset/zidan/0.png

            cv2.imwrite(img_path, face_roi)
            # Menyimpan gambar wajah ke file PNG

            count += 1
            # Menambah jumlah gambar yang tersimpan

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Menggambar kotak hijau di sekitar wajah

            cv2.putText(
                frame,
                f"{count}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            # Menampilkan nomor gambar di atas wajah

        cv2.imshow("Capture Faces", frame)
        # Menampilkan hasil kamera di window OpenCV

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Jika tombol 'q' ditekan, keluar dari loop
            break

        if count >= max_images:
            # Jika jumlah gambar sudah mencapai batas maksimal
            break

    cap.release()
    # Melepaskan akses webcam

    cv2.destroyAllWindows()
    # Menutup semua window OpenCV

    print(f"[INFO] Selesai! Tersimpan {count} foto di folder {save_path}")
    # Menampilkan informasi jumlah gambar yang berhasil disimpan


if __name__ == "__main__":
    # Mengecek apakah file ini dijalankan langsung
    capture_faces("zidan")
    # Memanggil fungsi capture_faces dengan nama "zidan"
