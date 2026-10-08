from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ASSETS = BASE.parent / "assets"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

# baca gambar
gbr = cv2.imread(str(ASSETS / "barcelona.png"))

# ubah jadi grayscale
gbr_gray = cv2.cvtColor(gbr, cv2.COLOR_BGR2GRAY)

# buat filter, harus dalam np.array
h = np.array([[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]])

# lakukan filtering/convolution
filtered = cv2.filter2D(gbr_gray, -1, h)

# simpan hasil
cv2.imwrite(str(OUTPUT / "gray.png"), gbr_gray)
cv2.imwrite(str(OUTPUT / "filtered.png"), filtered)

# tampilkan di layar
cv2.imshow('Aslinya:', gbr_gray)
cv2.imshow('Hasilnya:', filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
