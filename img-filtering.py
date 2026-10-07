from pathlib import Path
import cv2
import numpy as np

#baca gambar
BASE = Path(__file__).parent
gbr = cv2.imread(str(BASE / "img" / "Barcelona.png"))

#ubah jadi grayscale
gbr_gray = cv2.cvtColor(gbr, cv2.COLOR_BGR2GRAY)

#buat filter, harus dalam np.array
h = np.array([[1/9, 1/9, 1/9],[1/9, 1/9, 1/9],[1/9,
1/9, 1/9 ]])

#lakukan filtering/convolution
filtered = cv2.filter2D(gbr_gray,-1,h)

#tampilkan di layar
cv2.imshow('Aslinya:',gbr_gray)
cv2.imshow('Hasilnya:',filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
