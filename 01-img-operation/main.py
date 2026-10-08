from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent
ASSETS = BASE.parent / "assets"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

# baca gambar
gbr1 = cv2.imread(str(ASSETS / "barcelona.png"))
gbr2 = cv2.imread(str(ASSETS / "tutwuri.png"))

# menyamakan ukuran gambar dengan resize
gbr1 = cv2.resize(gbr1, (128, 128), interpolation=cv2.INTER_LINEAR)
gbr2 = cv2.resize(gbr2, (128, 128), interpolation=cv2.INTER_LINEAR)

# addition
gbr3 = gbr1 + gbr2
cv2.imwrite(str(OUTPUT / "addition.png"), gbr3)
cv2.imshow('addition', gbr3)

# substraction
gbr4 = gbr1 - gbr2
cv2.imwrite(str(OUTPUT / "substraction.png"), gbr4)
cv2.imshow('substraction', gbr4)

# invert
import numpy as np
gbr5 = np.ones(gbr1.shape) * 255
gbr5 = gbr5 - gbr1
cv2.imwrite(str(OUTPUT / "invert.png"), gbr5.astype("uint8"))
cv2.imshow('invert', gbr5.astype("uint8"))

cv2.waitKey(0)
cv2.destroyAllWindows()
