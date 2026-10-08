from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent
ASSETS = BASE.parent / "assets"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

img = cv2.imread(str(ASSETS / "barcelona.png"), cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

cv2.imwrite(str(OUTPUT / "edges.png"), edges)
cv2.imshow("hasil", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
