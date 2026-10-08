# 02 - Filtering Citra

## Tujuan
Memahami konvolusi dengan filter rata-rata (mean/box blur) pada citra grayscale.

## Teori singkat
- Citra diubah ke grayscale dulu supaya filtering 1 kanal.
- Kernel `3x3` berisi `1/9` = tiap piksel output adalah rata-rata 9 tetangganya.
- Efeknya: noise berkurang, tapi tepi jadi lebih blur.
- `cv2.filter2D(gbr_gray, -1, h)` menerapkan kernel `h` ke seluruh citra.

## Kode kunci
```python
h = np.array([[1/9]*3]*3)
filtered = cv2.filter2D(gbr_gray, -1, h)
```

## Cara run
```bash
python 02-img-filtering/main.py
```
Input: `../assets/barcelona.png`. Hasil: `output/gray.png`, `output/filtered.png`.

## Input vs Output
![gray](output/gray.png)
![filtered](output/filtered.png)
