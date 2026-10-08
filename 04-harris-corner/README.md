# 04 - Harris Corner Detection

## Tujuan
Menemukan sudut (corner) sebagai fitur penting untuk tracking dan matching.

## Teori singkat
- `cv2.cornerHarris(gray, 2, 3, 0.04)`: `blockSize=2`, `ksize=3`, `k=0.04`.
- Respons sudut di-dilate supaya titiknya terlihat, lalu threshold `0.01 * max` untuk memilih sudut kuat.
- Piksel sudut diwarnai merah `[0,0,255]`.

## Kode kunci
```python
dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
```

## Cara run
```bash
python 04-harris-corner/main.py
```
Input: `../assets/barcelona.png`. Hasil: `output/corners.png`.

## Input vs Output
![corners](output/corners.png)
