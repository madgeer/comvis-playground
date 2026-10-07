from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).parent
img = cv2.imread(str(BASE / "img" / "Barcelona.png"))
edges = cv2.Canny(img,100,200)
cv2.imshow("hasil",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()