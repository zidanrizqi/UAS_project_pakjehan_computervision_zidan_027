import cv2
import os

face_cascade = cv2.CascadeClassifier("../assets/haarcascade_frontalface_default.xml")

source = "raw_dataset"
dest = "../dataset"

for person in os.listdir(source):
    src_path = os.path.join(source, person)
    dst_path = os.path.join(dest, person)
    os.makedirs(dst_path, exist_ok=True)

    for img_name in os.listdir(src_path):
        # print(img_name)
        img_path = os.path.join(src_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = gray[y : y + h, x : x + w]
            save_path = os.path.join(dst_path, f"{img_name}")
            cv2.imwrite(save_path, face_crop)
