# 01 - Operasi Citra

## Tujuan
Memahami operasi piksel dasar: addition, substraction, dan invert (negatif).

## Teori singkat
- `addition (gbr1 + gbr2)`: menjumlahkan nilai piksel, berguna untuk menggabung dua citra.
- `substraction (gbr1 - gbr2)`: selisih piksel, berguna melihat perbedaan.
- `invert (255 - gbr1)`: membalik intensitas, area gelap jadi terang dan sebaliknya.
- Kedua gambar di-`resize` ke `128x128` dulu supaya ukurannya sama.

## Kode kunci
```python
gbr1 = cv2.resize(gbr1, (128, 128))
gbr3 = gbr1 + gbr2      # addition
gbr4 = gbr1 - gbr2      # substraction
gbr5 = 255 - gbr1       # invert
```

## Cara run
```bash
python 01-img-operation/main.py
```
Input diambil dari `../assets/barcelona.png` dan `../assets/tutwuri.png`.
Hasil disimpan otomatis di `output/`.

## Input vs Output
| Input | Output |
|---|---|
| `assets/barcelona.png` + `assets/tutwuri.png` | `output/addition.png`, `output/substraction.png`, `output/invert.png` |

![addition](output/addition.png)
