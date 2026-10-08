from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent
ASSETS = BASE.parent / "assets"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

img = cv2.imread(str(ASSETS / "presiden.jpeg"))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread(str(ASSETS / "template.png"))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

h, w = template_gray.shape

# metode yg bisa dipakai
metode = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
          cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# proses matching
res = cv2.matchTemplate(img_gray, template_gray, metode[0])
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# buat bounding box
bottom_right = (max_loc[0] + w, max_loc[1] + h)
cv2.rectangle(img_gray, max_loc, bottom_right, 255, 2)
cv2.rectangle(img, max_loc, bottom_right, (0, 0, 255), 2)

cv2.imwrite(str(OUTPUT / "match_gray.png"), img_gray)
cv2.imwrite(str(OUTPUT / "match_color.png"), img)

cv2.imshow('template', template)
cv2.imshow('hasil berwarna', img)
cv2.imshow('hasil hitam putih', img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
