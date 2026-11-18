# Laporan Perbandingan: BoVW (SIFT + K-Means + SVM) vs DNN (MLP)

**Kode yang dibandingkan**
- `comvis/09-deepl/BoVW/BoVW.py` — metode 1
- `comvis/09-deepl/DNN/DNN.py` — metode 2

**Tanggal pengujian:** 4 November 2026
**Dataset:** 297 citra, 3 kelas (`wl_cat`, `wl_deer`, `wl_horse`), 99 citra per kelas

> **Cara membaca laporan ini:** bagian 1–2 menjelaskan hasil dan cara kerja kedua program tanpa istilah rumit. Bagian 5 dan 6 adalah jawaban utama untuk pertanyaan **"kenapa metode 1 hasilnya begini dan kenapa metode 2 hasilnya begini"**. Bagian 8 membahas dua kesalahan metodologis yang membuat angka belum bisa dipercaya sepenuhnya.

---

## 1. Ringkasan yang Gampang Dicerna

### 1.1 Hasil dalam satu tabel

| Metrik (macro) | Metode 1: BoVW | Metode 2: DNN (MLP) | Selisih |
|---|---:|---:|---:|
| Accuracy | 0,4000 | **0,4333** | +3,33 pp |
| Precision | 0,4044 | **0,4815** | +7,71 pp |
| Recall | 0,4000 | **0,4333** | +3,33 pp |
| F1-Score | 0,3959 | **0,4404** | +4,44 pp |
| Benar / total | 24 / 60 | **26 / 60** | +2 citra |
| Waktu jalan (CPU) | **±8,9 s** | ±15,2 s | +6,3 s |

### 1.2 Empat kalimat inti

1. **DNN menang tipis di atas kertas**, tetapi kemenangannya hanya **2 citra dari 60**. Secara statistik ini **tidak signifikan** (McNemar p = 0,8506).
2. **Keduanya masih lemah.** Untuk 3 kelas, menebak acak saja sudah dapat 33,3%. BoVW 40% dan DNN 43% hanya sedikit di atas tebakan acak.
3. **Masalah utamanya bukan pemilihan classifier**, tetapi **representasi fitur**: BoVW membuang posisi objek lewat histogram, DNN membuang struktur gambar lewat `Flatten` pada citra 20×20.
4. Ada **dua cacat metodologis** yang harus diperbaiki dulu: **kebocoran data (data leakage) di BoVW** dan **bobot terbaik DNN tidak dipakai saat evaluasi**.

### 1.3 Analogi 30 detik

| Program | Analogi |
|---|---|
| **BoVW** | Mendeskripsikan lukisan dengan kalimat *"ada 30 goresan jenis A, 12 goresan jenis B, ..."* — **tanpa menyebut di mana letaknya**. Dua hewan berbulu mirip bisa menghasilkan daftar goresan yang mirip. |
| **DNN (MLP)** | Menerima **400 potongan puzzle yang sudah diacak**, lalu diminta menebak gambar apa. Potongan tetangga tidak lagi dikenali sebagai tetangga. |
| **Data leakage BoVW** | Kamus soal dibuat sambil **melihat soal ujian**. Nilai jadi terlihat lebih bagus dari kemampuan sebenarnya. |
| **Checkpoint DNN** | Jawaban terbaik disimpan di laci, tetapi yang dikumpulkan ke guru adalah **jawaban terakhir**. |

---

## 2. Kenalan: Apa yang Sebenarnya Dilakukan Tiap Program?

### 2.1 Metode 1 — BoVW (Bag of Visual Words)

BoVW adalah cara "mengubah gambar menjadi daftar kata". Idenya:

1. **SIFT** mencari titik-titik menarik di gambar (sudut, tekstur, bercak) dan mendeskripsikan tiap titik dengan 128 angka.
2. Semua deskriptor dari semua gambar dikumpulkan, lalu **K-Means** mengelompokkannya menjadi **100 "kata visual"** (codebook).
3. Setiap gambar diubah menjadi **histogram 100 bin**: bin ke-*i* berisi berapa kali kata visual ke-*i* muncul di gambar itu.
4. **SVM linear** belajar memisahkan histogram cat, deer, dan horse.

**Yang dilihat model:** *jenis tekstur apa saja yang ada*, bukan *bentuk hewannya* dan bukan *letak teksturnya*.

### 2.2 Metode 2 — DNN (Multi-Layer Perceptron)

Program ini disebut "DNN", tetapi arsitekturnya **MLP murni, bukan CNN**:

1. Gambar diperkecil menjadi **20×20 piksel grayscale** → 400 angka.
2. 400 angka itu **di-`Flatten`** menjadi satu baris, lalu masuk ke dua hidden layer berisi 64 neuron (ReLU).
3. Output 3 neuron dengan **softmax** → probabilitas cat / deer / horse.
4. Dilatih 10 epoch dengan Adam (lr = 0,01), batch 32.

**Yang dilihat model:** *pola intensitas piksel*, tetapi **tidak tahu piksel mana yang bertetangga** karena sudah diratakan.

### 2.3 Perbedaan inti kedua program

| Aspek | Metode 1: BoVW | Metode 2: DNN (MLP) |
|---|---|---|
| Cara mendapat fitur | **Dibuat manusia** (SIFT + K-Means) | **Dipelajari sendiri** dari piksel |
| Input | Grayscale, ukuran asli 80×120 / 120×80 | Grayscale, **dipaksa 20×20** |
| Representasi | Histogram 100 bin | 400 piksel mentah |
| Informasi posisi | Hilang (histogram tidak berurutan) | Hilang (di-`Flatten`) |
| Informasi bentuk | Tidak ada, hanya tekstur lokal | Tidak ada, hanya intensitas per piksel |
| Classifier | SVM linear | MLP 2 hidden layer (64, 64) |
| Normalisasi | `StandardScaler` pada histogram | `/255` pada piksel |
| Regularisasi | Tidak ada | **Tidak ada** (tanpa Dropout/BN/augmentasi) |
| Training | K-Means + SVM (sekali jalan) | 10 epoch, Adam lr = 0,01 |
| Evaluasi | Satu split 80/20 | Satu split 80/20 + 25% data latih untuk validasi |
| Kelebihan utama | Ringan, tahan terhadap skala/rotasi | Fleksibel, bisa belajar pola non-linear |
| Kelemahan utama | Bergantung pada kualitas codebook & normalisasi | Tidak cocok untuk gambar karena tidak ada konvolusi |

**Kesamaan penting:** keduanya **sama-sama membuang informasi spasial**, hanya dengan cara berbeda. Itu sebabnya performanya mirip dan sama-sama rendah.

---

## 3. Apakah Perbandingannya Adil?

Split data **identik**: 80% latih (237 citra) dan 20% uji (60 citra), stratified, `random_state=42`. Pengecekan indeks dan label uji menghasilkan `same split: True` dan `same labels: True` pada 60 sampel (20 per kelas). Jadi pada level **data uji**, perbandingan ini apple-to-apple.

Namun ada beberapa hal yang membuat **kondisinya tidak sepenuhnya setara**:

| Hal | BoVW | DNN | Efek |
|---|---|---|---|
| Resolusi | 80×120 (asli) | 20×20 (diperkecil) | DNN kehilangan jauh lebih banyak detail |
| Orientasi gambar | Dipertahankan | Diabaikan (resize persegi) | Dataset berisi 231 gambar landscape dan 66 portrait; DNN merusak rasio aspek keduanya |
| Kebocoran data | **Ada** (codebook dari seluruh dataset) | Tidak ada | Angka BoVW cenderung **terlalu optimistis** |
| Model terbaik | Langsung dipakai | **Tidak dipakai** | Angka DNN bukan performa terbaik yang tersimpan |
| Validasi | Tidak ada | `validation_split=0,25` (≈59 sampel) | Kurva validasi DNN berisik |

**Artinya:** meskipun data ujinya sama, BoVW mendapat "bocoran" sedangkan DNN tidak memakai model terbaiknya. Dua kelemahan ini arahnya berlawanan, sehingga kesimpulan "DNN sedikit lebih baik" tetap **tidak bisa dipercaya sebagai klaim kuat**.

---

## 4. Hasil Lengkap dan Cara Membacanya

### 4.1 Metrik keseluruhan

| Metrik | BoVW | DNN |
|---|---:|---:|
| Accuracy | 0,4000 | 0,4333 |
| Precision (macro) | 0,4044 | 0,4815 |
| Recall (macro) | 0,4000 | 0,4333 |
| F1 (macro) | 0,3959 | 0,4404 |
| Benar / total | 24 / 60 | 26 / 60 |

**Cara membaca:** dari 60 citra uji, BoVW benar 24 dan DNN benar 26. Selisih 2 citra. Dengan 3 kelas, menebak acak rata-rata benar 20 citra. Jadi BoVW hanya **4 citra di atas tebakan acak**, DNN **6 citra di atas tebakan acak**.

### 4.2 Metrik per kelas

**BoVW**

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — cat | 0,3846 | 0,5000 | 0,4348 | 20 |
| 1 — deer | 0,4000 | 0,4000 | 0,4000 | 20 |
| 2 — horse | 0,4286 | 0,3000 | 0,3529 | 20 |
| **Macro avg** | **0,4044** | **0,4000** | **0,3959** | 60 |

**DNN**

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — cat | 0,67 | 0,40 | 0,50 | 20 |
| 1 — deer | 0,33 | 0,50 | 0,40 | 20 |
| 2 — horse | 0,44 | 0,40 | 0,42 | 20 |
| **Macro avg** | **0,48** | **0,43** | **0,44** | 60 |

**Cara membaca:**
- BoVW **lebih merata**: F1 tiap kelas 0,35–0,43, tetapi tidak ada kelas yang benar-benar bagus.
- DNN **lebih tidak merata**: F1 0,40–0,50. Cat punya precision tinggi (0,67) tetapi recall rendah (0,40) — artinya *kalau DNN bilang cat, biasanya benar; tetapi banyak cat yang tidak dikenali*.
- **Horse paling sulit untuk BoVW** (recall 0,30: 14 dari 20 horse salah).
- **Deer paling sulit untuk DNN dari sisi precision** (0,33: 2 dari 3 tebakan "deer" salah).

### 4.3 Confusion matrix (baris = aktual, kolom = prediksi)

**BoVW** — akurasi 0,4000

```
            pred cat  pred deer  pred horse
true cat  [    10         6          4   ]
true deer [     8         8          4   ]
true horse[     8         6          6   ]
```

**DNN** — akurasi 0,4333

```
            pred cat  pred deer  pred horse
true cat  [     8        11          1   ]
true deer [     1        10          9   ]
true horse[     3         9          8   ]
```

**Yang langsung terlihat:**
- **Deer ↔ horse adalah pasangan paling sering tertukar di kedua model.** BoVW: 10 kesalahan silang. DNN: 18 kesalahan silang. Keduanya mamalia berkaki empat dengan siluet mirip, apalagi pada gambar kecil/abu-abu.
- **DNN sangat bias ke deer**: 11 dari 20 cat diprediksi deer. Model cenderung memilih deer saat ragu.
- **BoVW lebih seimbang**, tetapi horse sering diprediksi cat (8 dari 20).

### 4.4 Kurva training DNN (10 epoch)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---:|---:|---:|---:|---:|
| 1 | 0,2994 | 1,4926 | 0,3667 | 1,1040 |
| 2 | 0,4011 | 1,1113 | 0,3833 | 1,1930 |
| 3 | 0,4350 | 1,0800 | 0,3333 | 1,1204 |
| 4 | 0,4802 | 1,0361 | 0,3833 | 1,1039 |
| 5 | 0,5141 | 1,0077 | 0,4167 | 1,1359 |
| 6 | 0,5819 | 0,9566 | 0,4167 | 1,0988 |
| 7 | 0,6441 | 0,9015 | **0,4500** | 1,0860 |
| 8 | 0,6441 | 0,8523 | 0,4000 | 1,1147 |
| 9 | 0,6610 | 0,8371 | **0,4500** | **1,0829** |
| 10 | 0,6836 | 0,7864 | 0,4333 | 1,1683 |

**Cara membaca tanpa istilah rumit:**
- **Train accuracy naik terus** (0,30 → 0,68): model makin jago **menghafal** 178 gambar latih.
- **Val accuracy mandek** di 0,33–0,45 dan **val loss justru naik** di akhir (1,08 → 1,17): kemampuan pada data yang belum pernah dilihat **tidak ikut membaik**.
- Jarak train–val di epoch terakhir ≈ **25 pp**. Inilah tanda klasik **overfitting**: model menghafal, bukan memahami.
- Pada saat yang sama, val hanya 43% → model juga **terlalu lemah secara kapasitas/representasi**. Jadi bukan sekadar overfitting, tetapi **overfitting pada data kecil sekaligus underfitting secara kualitas fitur**.
- `val_accuracy` terbaik (0,45) ada di epoch 7 dan 9, sesuai nama checkpoint yang tersimpan (`weights-improv-07-0.45.keras`).

### 4.5 Uji signifikansi: apakah selisih 2 citra berarti?

| Kondisi | Jumlah |
|---|---:|
| Kedua model benar | 11 |
| Kedua model salah | 21 |
| Hanya BoVW benar | 13 |
| Hanya DNN benar | 15 |

- **McNemar exact test: p = 0,8506.**
- Interval kepercayaan 95% accuracy:
  - BoVW: **[0,286; 0,526]**
  - DNN: **[0,316; 0,559]**

**Artinya:** dari 28 citra tempat kedua model berbeda pendapat, BoVW benar 13 kali dan DNN 15 kali. Selisih 2 itu **wajar terjadi karena kebetulan**. Rentang kepercayaan keduanya juga tumpang tindih lebar. **DNN belum terbukti lebih baik dari BoVW** pada setup ini.

---

## 5. Metode 1 (BoVW): Kenapa Hasilnya Begini?

BoVW hanya mencapai 40%. Berikut faktor-faktornya, diurutkan dari yang paling berpengaruh.

### Faktor 1 — Histogram tidak dinormalisasi (dampak besar)

**Apa yang terjadi:** histogram berisi **jumlah mentah** kemunculan tiap kata visual. Citra dengan 27 keypoint menghasilkan total 27, citra dengan 176 keypoint menghasilkan total 176 — beda **6,5 kali**.

**Kenapa berpengaruh:** SVM linear mengambil keputusan berdasarkan jarak/dot product antar vektor. Citra bertekstur kaya (banyak keypoint) otomatis punya nilai besar di banyak bin, sehingga **mendominasi** perhitungan. Dua citra dengan objek sama tetapi jumlah keypoint berbeda akan terlihat jauh berbeda.

**Bukti:** rata-rata hanya **52,4 dari 100 bin** terisi (min 18, max 80). Histogram sangat sparse, dan `StandardScaler` hanya menyeragamkan skala **per bin**, bukan menghilangkan **bias jumlah total per gambar**.

**Dampak:** model sebagian belajar "gambar ini banyak teksturnya" alih-alih "gambar ini kucing".

### Faktor 2 — Codebook dibuat dari seluruh dataset (data leakage, dampak besar)

**Apa yang terjadi:**

```python
all_words = np.vstack(local_feature_list)   # descriptor dari SELURUH X
kmeans.fit(all_words)                       # termasuk descriptor data uji
```

**Kenapa berpengaruh:** kata visual (cluster center) dipilih dengan melihat descriptor dari data uji. Informasi test "bocor" ke proses yang seharusnya hanya memakai data latih.

**Dampak:** akurasi 40% **cenderung terlalu optimistis**. Jika codebook dibangun hanya dari data latih, performa bersihnya kemungkinan **lebih rendah lagi**. Justru inilah temuan penting: **meskipun ada kebocoran, BoVW tetap hanya 40%** — artinya representasinya memang lemah, bukan cuma soal leakage.

### Faktor 3 — Vocabulary terlalu kecil (k = 100) (dampak sedang–besar)

**Apa yang terjadi:** 25.987 descriptor dipaksa masuk ke hanya **100 kata**. Rata-rata ≈260 descriptor per kata.

**Kenapa berpengaruh:** kata visual menjadi sangat kasar. Tekstur bulu kucing, rusa, dan kuda bisa jatuh ke kata yang sama. Model tidak punya "kosakata" cukup untuk membedakan tekstur halus.

**Dampak:** batas antar kelas kabur, terutama deer vs horse.

### Faktor 4 — SIFT menangkap tekstur latar, bukan bentuk objek (dampak besar)

**Apa yang terjadi:** SIFT mencari sudut/tekstur lokal di **seluruh gambar**, termasuk rumput, daun, dan latar. Tidak ada mekanisme yang tahu mana "objek" dan mana "latar".

**Kenapa berpengaruh:** pada dataset hewan di alam, latar deer dan horse bisa mirip (rerumputan, semak). Model jadi memakai latar sebagai petunjuk, bukan bentuk hewan.

**Bukti:** deer ↔ horse menyumbang **10 kesalahan silang** di BoVW. Ini konsisten dengan "tekstur mirip, bentuk tidak terlihat".

### Faktor 5 — Informasi posisi hilang total (dampak sedang)

**Apa yang terjadi:** histogram bersifat *orderless* — urutan dan lokasi kata visual tidak disimpan.

**Kenapa berpengaruh:** kucing, rusa, dan kuda dibedakan terutama oleh **susunan bentuk** (kepala, kaki, tanduk, ekor). BoVW tidak bisa melihat susunan itu.

**Dampak:** dua gambar dengan tekstur sama tetapi susunan berbeda dianggap identik.

### Faktor 6 — Keputusan memakai SVM linear atas histogram mentah (dampak sedang)

**Kenapa berpengaruh:** SVM linear memisahkan data dengan satu bidang lurus. Untuk histogram sparse yang belum dinormalisasi, bidang ini mudah "tertipu" oleh bin dengan nilai ekstrem. Kernel non-linear atau normalisasi TF-IDF biasanya lebih cocok.

**Catatan:** `probability=True` dipakai padahal hanya `predict()` yang diperlukan. Ini tidak mengubah hasil, tetapi menambah komputasi dan memunculkan `FutureWarning` di scikit-learn 1.9.

### Faktor 7 — Evaluasi hanya satu split dengan 60 citra uji (dampak pada keandalan angka)

**Kenapa berpengaruh:** dengan n = 60, satu atau dua citra yang berubah bisa menggeser akurasi ±3,3 pp. Tanpa cross-validation, angka 40% bisa berubah cukup banyak jika split diulang.

**Dampak:** estimasi tidak stabil; perbandingan antar model jadi rapuh.

### Faktor 8 — Faktor sekunder

- **SIFT dijalankan dua kali per citra** (sekali untuk codebook, sekali untuk histogram) → boros ±50% waktu, tetapi **tidak** menurunkan akurasi.
- **Orientasi gambar tidak diseragamkan**: 231 gambar landscape 80×120 dan 66 gambar portrait 120×80. SIFT relatif tahan rotasi, jadi ini bukan penyebab utama, tetapi menambah variasi jumlah keypoint dan latar.
- **`get_bovw_hist` tidak menangani `descriptors is None`** → berpotensi crash pada gambar tanpa keypoint. Pada dataset ini semua gambar punya keypoint, jadi tidak terpicu.

### Rantai sebab-akibat BoVW

```
SIFT melihat seluruh gambar (termasuk latar)
        ↓
Histogram hanya menghitung jenis tekstur, tanpa posisi
        ↓
Jumlah keypoint berbeda 6,5× dan histogram tidak dinormalisasi
        ↓
Vektor gambar didominasi oleh "banyak/sedikitnya tekstur"
        ↓
SVM linear sulit memisahkan deer vs horse (tekstur & latar mirip)
        ↓
Akurasi 40% (hanya 4 citra di atas tebakan acak)
```

### Yang **bukan** penyebab utama

- **SIFT dihitung dua kali** → hanya masalah waktu, bukan akurasi.
- **`StandardScaler`** → sudah membantu menyeimbangkan skala per bin, tetapi tidak memperbaiki bias jumlah keypoint.
- **Jumlah data 297** → memang kecil, tetapi bukan satu-satunya sebab; representasi yang lemah memperparah.

---

## 6. Metode 2 (DNN): Kenapa Hasilnya Begini?

DNN hanya mencapai 43%. Berikut faktor-faktornya, diurutkan dari yang paling berpengaruh.

### Faktor 1 — Ini MLP, bukan CNN; `Flatten` membuang struktur gambar (dampak paling besar)

**Apa yang terjadi:**

```python
inputs = Input(shape=(20, 20))
flatten = Flatten()(inputs)          # 20×20 → 400 angka berurutan
h_layer = Dense(64, activation="relu")(flatten)
```

**Kenapa berpengaruh:** setelah `Flatten`, model **tidak tahu piksel mana yang bertetangga**. Padahal ciri visual paling dasar — garis, sudut, kontur — justru lahir dari piksel yang berdekatan. MLP harus mempelajari hubungan itu dari nol, dan dengan hanya 178 gambar latih itu tidak mungkin.

**Bukti:** model punya **30.019 parameter** untuk 237 gambar latih. Tanpa *inductive bias* bahwa gambar itu dua dimensi, model cenderung menghafal pola piksel, bukan belajar bentuk.

**Dampak:** ini kelemahan **struktural**. Bahkan dengan training sempurna, MLP pada piksel mentah sulit menandingi CNN untuk klasifikasi gambar.

### Faktor 2 — Input 20×20 terlalu kecil (dampak besar)

**Apa yang terjadi:** semua gambar diperkecil ke 20×20 (400 piksel), termasuk yang aslinya 80×120 atau 120×80.

**Kenapa berpengaruh:** detail pembeda hewan — bentuk tanduk, moncong, telinga, pola bulu — hilang. Selain itu, `cv2.resize(img, (20, 20))` **merusak rasio aspek**: gambar landscape dan portrait dipaksa menjadi persegi.

**Dampak:** informasi yang tersedia sebelum model belajar sudah sangat sedikit. Di titik ini, sebagian besar kesalahan sudah "ditakdirkan" oleh preprocessing.

### Faktor 3 — Tidak ada regularisasi (dampak besar)

**Apa yang terjadi:** tidak ada Dropout, BatchNorm, L2, maupun augmentasi data.

**Kenapa berpengaruh:** dengan 178 gambar latih dan 30.019 parameter, model punya cukup kapasitas untuk **menghafal** data latih.

**Bukti dari kurva training:** train accuracy naik ke 0,68, val accuracy mandek di 0,43, dan val loss **naik** di akhir. Gap ≈25 pp.

**Dampak:** overfitting. Model bagus di data latih, buruk di data baru.

### Faktor 4 — Learning rate 0,01 untuk Adam terlalu besar, dan hanya 10 epoch (dampak sedang)

**Kenapa berpengaruh:** Adam dengan lr = 0,01 sering membuat langkah training terlalu agresif. Buktinya, **val loss naik-turun** (1,10 → 1,19 → 1,10 → 1,17) sementara train loss turun mulus. Model tidak pernah benar-benar stabil.

**Dampak:** konvergensi tidak mulus. Di sisi lain, 10 epoch juga terlalu sedikit untuk model belajar dengan tenang. Jadi ada dua masalah sekaligus: **langkah terlalu besar** dan **waktu belajar terlalu singkat**.

### Faktor 5 — Bobot terbaik dari checkpoint tidak pernah dipakai (dampak besar pada validitas angka)

**Apa yang terjadi:**

```python
checkpoint = ModelCheckpoint(..., save_best_only=True, mode='max')
...
model.fit(...)
output = model.predict(X_test)   # memakai bobot epoch TERAKHIR
```

**Kenapa berpengaruh:** `ModelCheckpoint` sudah menyimpan bobot dengan `val_accuracy` terbaik (0,45 di epoch 7 dan 9), tetapi `model` di memori tetap berisi bobot **epoch ke-10** (val 0,4333). Bobot terbaik tidak pernah di-`load`.

**Dampak:** angka 43,33% yang dilaporkan **bukan** performa model terbaik yang sebenarnya tersimpan. Secara metodologis ini keliru, meskipun selisihnya kecil (0,45 vs 0,4333 pada validasi).

### Faktor 6 — Validasi tidak stratified dan terlalu kecil (dampak pada keandalan angka)

**Apa yang terjadi:** `validation_split=0,25` mengambil **25% terakhir** dari data latih (≈59 sampel). Data latih sudah di-shuffle, jadi komposisinya kira-kira acak, tetapi **tidak dijamin stratified**.

**Kenapa berpengaruh:** dengan ±59 sampel validasi, satu kelas bisa sedikit lebih banyak/lebih sedikit. Val accuracy jadi berisik: 0,3667 → 0,3833 → 0,3333 → 0,3833 → ...

**Dampak:** checkpoint "terbaik" bisa terpilih karena keberuntungan sampel validasi, bukan karena model benar-benar membaik.

### Faktor 7 — Model condong menebak "deer" (dampak pada pola kesalahan)

**Bukti:** 11 dari 20 citra cat diprediksi deer; precision deer hanya 0,33.

**Kenapa bisa terjadi:** pada dataset kecil, softmax + cross-entropy mudah jatuh ke solusi "satu kelas menyerap tebakan ragu". Kelas deer tampaknya menjadi "kelas default" model. Ini bukan bug kode, melainkan gejala model yang belum punya fitur pembeda cukup.

### Faktor 8 — Grayscale menghilangkan isyarat warna (dampak sedang)

**Apa yang terjadi:** `MyUtils.load(..., mode=0)` membaca gambar sebagai grayscale.

**Kenapa berpengaruh:** warna adalah isyarat kuat untuk membedakan hewan (misalnya pola bulu). Menghilangkan warna mempersulit tugas, terutama pada resolusi 20×20 yang sudah minim detail.

### Rantai sebab-akibat DNN

```
Citra diperkecil ke 20×20 dan dijadikan grayscale
        ↓
Flatten mengubah gambar menjadi 400 angka tanpa relasi tetangga
        ↓
MLP 2×64 neuron harus menghafal pola dari 178 gambar latih
        ↓
Tanpa Dropout/BN/augmentasi + lr 0,01 → train naik, val mandek
        ↓
Overfitting (gap 25 pp) sekaligus representasi terlalu lemah
        ↓
Akurasi 43% dan bias menebak "deer"
```

### Yang **bukan** penyebab utama

- **Jumlah epoch 10** → bukan satu-satunya sebab; menambah epoch tanpa regularisasi hanya memperparah overfitting.
- **`/255`** → sudah benar dan membantu konvergensi.
- **Arsitektur 2 hidden layer** → bukan masalah utama; masalahnya adalah **tidak ada konvolusi** dan input terlalu kecil.
- **`LabelBinarizer` + softmax** → sudah tepat untuk 3 kelas.

---

## 7. Perbedaan Pola Perilaku Kedua Model

Angka akhirnya mirip, tetapi **cara keduanya salah berbeda**.

| Aspek | BoVW | DNN (MLP) |
|---|---|---|
| Sumber kesalahan utama | Fitur tekstur tanpa posisi + histogram tidak normal | Tidak ada konvolusi + input 20×20 + overfitting |
| Distribusi kesalahan | Relatif merata antar kelas | Sangat bias ke deer |
| Kelas terburuk | Horse (recall 0,30) | Cat (recall 0,40) dan deer (precision 0,33) |
| Kesalahan deer ↔ horse | 10 citra | 18 citra |
| Sensitivitas terhadap latar | Tinggi (SIFT menangkap latar) | Sedang (piksel latar ikut masuk) |
| Sensitivitas terhadap jumlah tekstur | Tinggi (raw count) | Tidak relevan |
| Stabilitas hasil | Ditentukan codebook & K-Means | Ditentukan inisialisasi & urutan batch (TF tidak deterministik penuh) |
| Biaya komputasi | ±8,9 s (didominasi SIFT) | ±15,2 s (didominasi training TF) |
| Cocok untuk | Dataset tekstur, citra medis, retrieval | Data tabular; **bukan** gambar tanpa CNN |

**Poin penting:** BoVW "lebih seimbang" bukan karena lebih pintar, tetapi karena semua kelas diperlakukan dengan cara yang sama (histogram bersama). DNN "lebih polar" karena softmax bisa jatuh ke solusi yang menyerap satu kelas.

---

## 8. Dua Masalah Metodologis yang Wajib Diperbaiki

### 8.1 BoVW — Data leakage (severity: tinggi)

```python
all_words = np.vstack(local_feature_list)   # dari SELURUH X, termasuk test
kmeans.fit(all_words)
```

**Masalah:** codebook dibangun memakai descriptor dari data uji.
**Akibat:** informasi test bocor ke tahap "belajar representasi"; akurasi 0,40 **cenderung optimistis**.
**Perbaikan:** bangun codebook **hanya dari descriptor data latih**, lalu gunakan codebook itu untuk mengubah data latih dan uji.

### 8.2 DNN — Best checkpoint tidak dipakai (severity: tinggi)

```python
checkpoint = ModelCheckpoint(..., save_best_only=True, mode='max')
...
output = model.predict(X_test)   # bobot epoch terakhir, bukan terbaik
```

**Masalah:** bobot terbaik disimpan ke file, tetapi tidak pernah dimuat kembali.
**Akibat:** evaluasi memakai epoch ke-10, bukan model terbaik (val 0,45).
**Perbaikan:** `model = load_model(best_path)` sebelum `predict`, atau gunakan `EarlyStopping(restore_best_weights=True)`.

### 8.3 Perbandingan belum sepenuhnya setara

- BoVW melihat citra 80×120; DNN hanya 20×20 → DNN kekurangan informasi.
- BoVW mendapat keuntungan dari kebocoran codebook → BoVW terlalu diuntungkan.
- Keduanya hanya satu split, tanpa cross-validation → variansi estimasi tinggi (n uji = 60).

---

## 9. Rekomendasi: Apa yang Harus Dilakukan dan Kenapa

### Prioritas 1 — Perbaiki validitas eksperimen

| Masalah | Tindakan | Kenapa |
|---|---|---|
| Codebook BoVW bocor | Fit K-Means hanya pada descriptor data latih | Agar akurasi mencerminkan kemampuan sebenarnya |
| Bobot terbaik DNN tidak dipakai | `load_model(best_path)` atau `EarlyStopping(restore_best_weights=True)` | Agar yang dievaluasi benar-benar model terbaik |
| Satu split, n = 60 | 5-fold stratified cross-validation, laporkan mean ± std | Agar kesimpulan tidak bergantung pada satu pembagian data |
| Perbandingan per sampel | Simpan prediksi tiap sampel | Agar bisa diuji McNemar/paired test |

### Prioritas 2 — Perbaiki representasi fitur

| Metode | Tindakan | Kenapa |
|---|---|---|
| BoVW | Normalisasi histogram L1/L2 + TF-IDF atau power normalization (akar) | Menghilangkan bias jumlah keypoint |
| BoVW | Tuning `k ∈ {50, 100, 200, 400}` dan hyperparameter SVM (`C`, kernel) via GridSearchCV | Vocabulary 100 terlalu kasar; belum pernah di-tuning |
| DNN | Ganti ke **CNN kecil** (2–3 blok Conv + Pool) pada 64×64 | Konvolusi memanfaatkan struktur 2D gambar |
| DNN | Tambahkan augmentasi (flip, crop, brightness) + Dropout/BN | Mengurangi overfitting pada 178 gambar latih |
| DNN | Turunkan lr ke 1e-3, tambah `ReduceLROnPlateau`, epoch 50–100 | Konvergensi lebih stabil |

### Prioritas 3 — Efisiensi dan kualitas kode

1. Cache descriptor SIFT (hilangkan perhitungan ganda).
2. Hapus import tak terpakai (`prometheus_client.Histogram`, `LabelBinarizer` di BoVW) dan opsi `probability=True`.
3. Tambahkan `classification_report` dan pemetaan nama kelas pada confusion matrix.
4. Pastikan folder `output/` dibuat otomatis (`os.makedirs(..., exist_ok=True)`).
5. Set `TF_DETERMINISTIC_OPS=1` dan `tf.config.experimental.enable_op_determinism()` agar hasil lebih reproducible.

### Target realistis

Dengan hanya 297 citra, akurasi >80% **sulit** dicapai tanpa transfer learning. Opsi paling menjanjikan:

- **Transfer learning** (MobileNetV2 / EfficientNetB0 pre-trained, fine-tuning ringan), atau
- **CNN kecil + augmentasi kuat + cross-validation**.

---

## 10. Kesimpulan Sederhana

1. **DNN (MLP) menang nominal** di semua metrik: accuracy 0,4333 vs 0,4000; F1 macro 0,4404 vs 0,3959.
2. **Kemenangan itu tidak signifikan**: hanya 2 citra dari 60, McNemar p = 0,8506, dan CI 95% saling tumpang tindih. Belum bisa disebut lebih baik.
3. **Keduanya masih jauh dari memadai**: 40–43% dibanding tebakan acak 33,3%. Akar masalahnya ada pada **representasi fitur dan kapasitas model**, bukan pada pemilihan classifier.
4. **BoVW** unggul dalam kesederhanaan dan biaya, tetapi kalah karena histogram tidak dinormalisasi, vocabulary kecil, SIFT menangkap latar, dan ada **data leakage** yang membuat angkanya terlihat lebih bagus dari seharusnya.
5. **DNN** lebih fleksibel, tetapi implementasinya adalah MLP pada citra 20×20 tanpa regularisasi. Struktur spasial hilang, overfitting terjadi, dan **checkpoint terbaik tidak dievaluasi**.
6. **Langkah perbaikan paling berdampak:**
   - DNN: **CNN + resolusi lebih tinggi + augmentasi**.
   - BoVW: **normalisasi histogram + codebook bebas leakage + tuning**.
   - Keduanya: **cross-validation** agar angka bisa dipercaya.

---

## 11. Lampiran

### 11.1 Lingkungan

- Python 3.13.1
- OpenCV 5.0.0
- scikit-learn 1.9.0
- TensorFlow 2.21.0
- CPU (tanpa GPU)

### 11.2 Perintah reproduksi

```bash
# BoVW
cd comvis/09-deepl/BoVW
python BoVW.py

# DNN
cd comvis/09-deepl/DNN
python DNN.py
```

### 11.3 Catatan reproduksi

- Kedua skrip memakai `random_state=42` / `SEED=42`.
- Hasil di laporan ini berasal dari eksekusi ulang dan identik dengan artefak yang ada (checkpoint `weights-improv-07-0.45.keras`).
- Saat BoVW dijalankan dengan scikit-learn 1.9, muncul `FutureWarning` terkait `SVC(probability=True)`; tidak memengaruhi hasil, tetapi sebaiknya diperbaiki.
- Pengukuran waktu bersifat indikatif (satu kali jalan, CPU, termasuk startup runtime).

### 11.4 Ringkasan angka mentah

```
BoVW  : acc=0.4000  P=0.4044  R=0.4000  F1=0.3959  (24/60)
DNN   : acc=0.4333  P=0.4815  R=0.4333  F1=0.4404  (26/60)
McNemar exact p = 0.8506
CI95 BoVW = [0.286, 0.526] ; CI95 DNN = [0.316, 0.559]
```

### 11.5 Temuan rinci pada kode

#### `BoVW.py`

| No | Temuan | Dampak | Saran |
|---|---|---|---|
| 1 | Codebook K-Means dilatih pada seluruh dataset (train + test) | **Data leakage**, akurasi optimistis | Fit K-Means hanya pada descriptor data latih |
| 2 | SIFT dihitung dua kali per citra | Boros ±50% waktu | Simpan `descriptors` sekali, pakai ulang |
| 3 | Histogram tidak dinormalisasi (raw count) | Bias terhadap citra dengan banyak keypoint (27–176) | L1/L2 normalization atau TF-IDF |
| 4 | `get_bovw_hist` tidak menangani `descriptors is None` | Rentan crash pada citra tanpa keypoint | Guard `if descriptors is None: return zeros` |
| 5 | `SVC(probability=True)` tidak dipakai dan deprecated di sklearn 1.9 | Warning + komputasi ekstra | Hapus `probability=True` |
| 6 | Import `prometheus_client.Histogram` dan `LabelBinarizer` tidak terpakai | Dependensi tak perlu, berpotensi error | Hapus |
| 7 | Tidak ada `classification_report` per kelas | Perilaku tiap kelas tidak terlihat | Tambahkan |
| 8 | k=100, kernel, dan C tidak pernah di-tuning | Performa suboptimal | GridSearchCV pada data latih |
| 9 | Hanya satu split, tanpa cross-validation | Estimasi tidak stabil | 5-fold stratified CV |

#### `DNN.py`

| No | Temuan | Dampak | Saran |
|---|---|---|---|
| 1 | Bobot terbaik dari checkpoint tidak di-*load* | Evaluasi memakai epoch terakhir, bukan model terbaik | `load_model(best_path)` atau `EarlyStopping(restore_best_weights=True)` |
| 2 | Arsitektur MLP, bukan CNN | Tidak ada invariansi spasial; cocok untuk tabular, buruk untuk citra | Ganti ke `Conv2D` + `MaxPooling2D` |
| 3 | Input 20×20 | Detail objek hilang | 64×64 atau 96×96 |
| 4 | `lr=0.01` untuk Adam terlalu besar; hanya 10 epoch | Konvergensi tidak stabil | `lr=1e-3`, `ReduceLROnPlateau`, epoch 50–100 |
| 5 | Tanpa Dropout/BatchNorm/L2/augmentasi | Overfitting (gap train–val 25 pp) | Tambahkan regularisasi + augmentasi |
| 6 | `validation_split=0.25` mengambil 25% terakhir data latih | Tidak eksplisit stratified | Split validasi manual dengan `train_test_split(stratify=...)` |
| 7 | `TrainingPlot` menulis ke `output/` tanpa `makedirs` | Crash bila folder belum ada | `os.makedirs('output', exist_ok=True)` |
| 8 | `H` tidak dipakai | Minor | Hapus |
| 9 | Label confusion matrix berupa angka 0/1/2 | Sulit dibaca | Petakan ke nama kelas |
| 10 | Determinisme TF tidak penuh | Hasil bisa bergeser antar run/device | Set `TF_DETERMINISTIC_OPS=1`, `tf.config.experimental.enable_op_determinism()` |
