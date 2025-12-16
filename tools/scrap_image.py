import requests
from bs4 import BeautifulSoup
import os

url = "https://www.1zoom.me/en/s/Celebrities-Jennifer_Lawrence/t2/1/"
save_dir = "raw_dataset/JenniferLawrence"

os.makedirs(save_dir, exist_ok=True)

# Ambil HTML
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")

# Cari gambar thumbnail (format situs 1zoom)
images = soup.select("img")

count = 0
for img in images:
    src = img.get("src")
    # if src and "/big/" not in src:  # skip template images
    try:
        img_data = requests.get(src).content
        filename = os.path.join(save_dir, f"jl_{count}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
        print("saved:", filename)
        count += 1
    except:
        pass

print("selesai. total:", count)
