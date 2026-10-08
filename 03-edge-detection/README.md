# 03 - Edge Detection (Canny)

## Tujuan
Mendeteksi tepi objek dengan algoritma Canny.

## Teori singkat
Canny bekerja 4 tahap: Gaussian blur -> gradien Sobel -> non-maximum suppression -> hysteresis thresholding.
- `threshold1=100`: batas bawah, `threshold2=200`: batas atas.
- Naikkan threshold kalau banyak tepi palsu, turunkan kalau tepi putus-putus.

## Kode kunci
```python
img = cv2.imread("barcelona.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)
```

## Cara run
```bash
python 03-edge-detection/main.py
```
Input: `../assets/barcelona.png`. Hasil: `output/edges.png`.

## Input vs Output
![edges](output/edges.png)
