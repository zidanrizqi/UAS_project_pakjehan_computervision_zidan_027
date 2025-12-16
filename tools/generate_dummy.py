import os

persons = [
    "johny_depp",
    "emma_watson",
    "angelina_jolly",
    "scarlett_johansson",
    "hugh_jackman",
]

base = "../dataset"
os.makedirs(base, exist_ok=True)

for p in persons:
    folder = os.path.join(base, p)
    os.makedirs(folder, exist_ok=True)

    for i in range(1, 6):
        # membuat file kosong dummy (nanti bisa ganti dengan gambar asli)
        open(os.path.join(folder, f"{p}_{i}.jpg"), "a").close()

print("Dummy dataset created.")
