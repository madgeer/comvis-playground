from pathlib import Path
import cv2
import numpy as np

#baca gambar
BASE = Path(__file__).parent
gbr1 = cv2.imread(str(BASE / "img" / "Barcelona.png"))
gbr2 = cv2.imread(str(BASE / "img" / "Tutwuri.png"))

#menyamakan ukuran gambar dengan resize
gbr1=cv2.resize(gbr1,(128,128),interpolation=cv2.INTER_LINEAR)
gbr2=cv2.resize(gbr2,(128,128),interpolation=cv2.INTER_LINEAR)

#addition
gbr3 = gbr1+gbr2
cv2.imshow('addition',gbr3)

#substraction
gbr4 = gbr1-gbr2
cv2.imshow('substraction',gbr4)

#invert
gbr5 = np.ones(gbr1.shape)*255
gbr5 = gbr5-gbr1
cv2.imshow('invert',gbr5)

cv2.waitKey(0)
cv2.destroyAllWindows()