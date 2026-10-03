# Transfer Learning: Klasifikasi Citra huruf_angka vs panah

## 1. Dataset

- Total 359 citra, 2 kelas: `huruf_angka` (180) dan `panah` (179).
- Pembagian mengikuti kolom `split` pada `metadata.csv` (tidak diacak ulang):

| Kelas | Train | Val | Test |
|---|---|---|---|
| huruf_angka | 100 | 29 | 51 |
| panah | 95 | 52 | 32 |
| **Total** | **195** | **81** | **83** |

- Kelas hampir seimbang (rasio ~1:1), sehingga class weight (inverse frequency) bernilai sekitar 0,98 dan 1,03 dan pengaruhnya kecil.
- Pemeriksaan kebocoran data: jumlah `session_id` yang muncul di lebih dari satu split = **0**, sehingga foto dari sesi yang sama tidak bocor antar split.
- Distribusi kondisi cahaya per kelas:

| Kelas | Gelap | Redup | Terang |
|---|---|---|---|
| huruf_angka | 3 | 43 | 134 |
| panah | 3 | 41 | 135 |

## 2. Metode

- Backbone MobileNetV3-Large (bobot ImageNet), head diganti 2 kelas.
- Input 224×224, augmentasi pada train (rotasi ±10°, translasi, skala, color jitter).
- 10 epoch, optimizer AdamW, seed 42. Bobot terbaik dipilih berdasarkan **val macro-F1**, bukan test.
- Tiga mode:
  1. `feature_extraction`: backbone dibekukan, hanya classifier dilatih.
  2. `fine_tune_sebagian`: blok `features[13:]` dan classifier dilatih.
  3. `fine_tune_penuh`: semua layer dilatih.

## 3. Hasil

| Mode | Param dilatih | Val macro-F1 | Test acc | Test macro-F1 | Waktu train (s) |
|---|---|---|---|---|---|
| feature_extraction | 1.232.642 | 0,9150 | 0,988 | 0,9874 | 19,4 |
| fine_tune_sebagian | 3.412.106 | 1,0000 | 1,000 | 1,0000 | 18,1 |
| fine_tune_penuh | 4.204.594 | 0,9865 | 1,000 | 1,0000 | 21,2 |

Mode terpilih berdasarkan val macro-F1: **fine_tune_sebagian**.

Latensi inferensi (1 citra, 224×224, rata-rata 100 kali setelah warm-up):

| Perangkat | Latensi mean (ms) | Std (ms) |
|---|---|---|
| GPU (T4) | 6,67 | 0,92 |
| CPU | 21,13 | 2,39 |

File pendukung: `hasil_3_mode.csv`, `latensi.csv`, `akurasi_per_epoch.png`, `cm_<mode>.png`.

## 4. Analisis

- Ketiga mode mencapai test acc 0,988–1,000. Dengan test set hanya 83 citra, selisih 1 citra salah (feature_extraction) tidak cukup untuk menyimpulkan satu mode jauh lebih baik.
- Fine-tuning (sebagian atau penuh) unggul di val macro-F1 dibanding feature extraction (0,915), yang menunjukkan adaptasi backbone membantu, meski perbedaannya perlu dibaca hati-hati karena data kecil.
- Kurva akurasi validasi tidak stabil, misalnya `fine_tune_sebagian` turun ke sekitar 0,55 pada epoch 9 lalu naik ke 1,0 pada epoch 10. Penyebab yang mungkin: train set kecil (195 citra) dan learning rate backbone yang relatif besar. Skor terbaik sebagian dapat dipengaruhi faktor kebetulan.
- Akurasi train lebih rendah daripada val/test karena dihitung dengan augmentasi aktif, bukan tanda underfitting.
- Latensi CPU (±21 ms) masih cukup cepat untuk inferensi waktu nyata pada citra tunggal; GPU sekitar 3 kali lebih cepat.

## 5. Keterbatasan

- Test set kecil (83 citra), sehingga angka 100% tidak berarti model sempurna di data baru.
- Hasil berasal dari satu seed. Untuk klaim yang lebih kuat, ulangi dengan beberapa seed dan laporkan rata-rata ± simpangan baku.
- Sebagian besar citra diambil pada cahaya terang (sekitar 75%), sedangkan cahaya gelap hanya 3 citra per kelas. Performa pada kondisi gelap tidak dapat dinilai dari data ini.

## 6. Struktur Repository dan Cara Menjalankan Ulang

```
.
├── README.md
├── dataset_raw.zip            # dataset (359 citra + metadata.csv)
├── transfer_learning.ipynb    # notebook Colab (kode training dan evaluasi)
└── hasil/
    ├── hasil_3_mode.csv
    ├── latensi.csv
    ├── akurasi_per_epoch.png
    ├── cm_feature_extraction.png
    ├── cm_fine_tune_sebagian.png
    ├── cm_fine_tune_penuh.png
    └── model_*.pth            # bobot terbaik tiap mode
```

Cara menjalankan ulang:

1. Buka `transfer_learning.ipynb` di Google Colab, pilih runtime GPU (T4).
2. Unggah `dataset_raw.zip` ke Google Drive, folder `transfer learning`.
3. Jalankan sel dari atas ke bawah. Hasil akan tersimpan di folder `hasil`.
