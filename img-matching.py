from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).parent
img = cv2.imread(str(BASE / "img" / "presiden.jpeg"))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread(str(BASE / "img" / "template.png"))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

ordo = template_gray.shape
w = ordo[0]
h = ordo[1]

# metode yg bisa dipakai
metode = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

#proses matching
res = cv2.matchTemplate(img_gray,template_gray,metode[0])
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#buat bounding box
cv2.rectangle(img_gray,max_loc, (max_loc[0]+w, max_loc[1]+h),255, 2)
cv2.rectangle(img,max_loc,(max_loc[0]+w, max_loc[1]+h),(0,0,255), 2)
cv2.imshow('template',template)
cv2.imshow('hasil berwarna',img)
cv2.imshow('hasil hitam putih',img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
