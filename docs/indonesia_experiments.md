# Protokol eksperimen tambahan kohor Indonesia

Dokumen ini menyiapkan analisis tambahan yang direncanakan dalam `Narration.md`. Kohor Indonesia belum tersedia; perintah di bawah belum menghasilkan temuan Indonesia. Semua tabel tingkat sampel dan keluaran baru disimpan lokal di `results/indonesia/`, yang diabaikan Git.

## Urutan analisis

1. **Bekukan model sumber sebelum melihat luaran Indonesia.** Jalankan `freeze` pada enam kohor sumber (`all`) dan empat kohor Asia (`asia`: Thailand dan Tiongkok). Setiap model memilih representasi dengan validasi internal berkelompok menurut BioProject, menentukan ambang dari prediksi *out of fold*, lalu menyimpan model serta hash berkas sumber. Dua model ini dievaluasi pada keseluruhan kohor Indonesia tanpa pelatihan ulang.
2. **Periksa masukan Indonesia.** Pastikan tabel hitungan genus berasal dari pipeline 16S rRNA yang terdokumentasi, label kasus dan kontrol didefinisikan secara klinis, ID sampel unik dan cocok persis, dan informasi platform, wilayah amplikon, batch, usia, jenis kelamin, BMI, serta kovariat lain dicatat bila tersedia. `check` memvalidasi struktur tabel; kesetaraan definisi MASLD dan keputusan inklusi tetap memerlukan pemeriksaan peneliti.
3. **Validasi eksternal utama.** `external` menerapkan model yang sudah dibekukan pada seluruh kohor Indonesia. Keluaran mencakup AUROC, AUPRC, proporsi kasus sebagai garis dasar AUPRC, interval bootstrap, akurasi seimbang, sensitivitas, spesifisitas, Brier score, ringkasan kalibrasi, dan prediksi per sampel. Hasil ini terpisah dari analisis pelatihan yang menggunakan data Indonesia.
4. **Profil, geografi, dan replikasi.** `profile` menghasilkan sepuluh genus paling melimpah, indeks Shannon dan Gini-Simpson, ordinasi Bray-Curtis/PCoA, dan PERMANOVA diagnosis dalam kohor Indonesia yang bersifat deskriptif tanpa penyesuaian kovariat. `geography` membuat ordinasi gabungan enam kohor sumber dan Indonesia dengan label negara, kohor, serta diagnosis. Negara dan kohor dapat saling terkonfounded sehingga ordinasi ini hanya deskriptif. `replicate` menguji arah serta besaran efek tujuh kandidat genus yang sudah dipilih dari hasil enam kohor, dengan daftar tetap pada `config/pre_indonesia_meta_analysis_taxa.csv`.
5. **Perbandingan pelatihan sekunder.** Setelah pembagian latih/uji Indonesia ditetapkan sebelum melihat hasil uji, `compare` melatih dua model: Indonesia saja dan Indonesia ditambah empat kohor Asia. Pemilihan fitur, representasi, dan ambang dilakukan pada data latih masing-masing; keduanya dinilai pada set uji Indonesia yang identik. Analisis ini bukan validasi eksternal dari model sumber yang dibekukan.

## Bentuk masukan

- `--otu`: TSV dengan fitur pada kolom pertama dan ID sampel pada kolom berikutnya. Nilai harus berupa hitungan nonnegatif, bukan proporsi. Gunakan `--table-format taxonomy` untuk anotasi taksonomi yang mengandung prefiks `g__` atau `--table-format genus` untuk nama genus. Fitur dengan nama sama dijumlahkan.
- `--metadata`: CSV dengan kolom ID sampel, diagnosis, dan bila perlu kolom pembagian latih/uji. Setiap sampel pada tabel kelimpahan harus memiliki tepat satu baris metadata. Nilai label ditentukan secara eksplisit lewat argumen CLI.
- Gunakan ID pseudonim tanpa informasi pasien. Berkas masukan Indonesia sebaiknya berada di folder lokal yang tidak dilacak Git. Jangan masukkan data terbatas ke repositori publik.

Contoh di bawah memakai nama kolom ilustratif. Sesuaikan dengan data yang benar ketika tersedia.

```bash
.venv/bin/python -m src.indonesia_extension freeze --source-sets all asia --model SVM

.venv/bin/python -m src.indonesia_extension check \
  --otu OTU-table/indonesia_genus.tsv --metadata OTU-table/indonesia_metadata.csv \
  --table-format genus --sample-column sample_id --label-column diagnosis \
  --case-value MASLD --control-value Control

.venv/bin/python -m src.indonesia_extension external \
  --otu OTU-table/indonesia_genus.tsv --metadata OTU-table/indonesia_metadata.csv \
  --table-format genus --sample-column sample_id --label-column diagnosis \
  --case-value MASLD --control-value Control \
  --locks results/indonesia/all_svm.lock.json results/indonesia/asia_svm.lock.json

.venv/bin/python -m src.indonesia_extension profile \
  --otu OTU-table/indonesia_genus.tsv --metadata OTU-table/indonesia_metadata.csv \
  --table-format genus --sample-column sample_id --label-column diagnosis \
  --case-value MASLD --control-value Control

.venv/bin/python -m src.indonesia_extension geography \
  --otu OTU-table/indonesia_genus.tsv --metadata OTU-table/indonesia_metadata.csv \
  --table-format genus --sample-column sample_id --label-column diagnosis \
  --case-value MASLD --control-value Control

.venv/bin/python -m src.indonesia_extension replicate \
  --otu OTU-table/indonesia_genus.tsv --metadata OTU-table/indonesia_metadata.csv \
  --table-format genus --sample-column sample_id --label-column diagnosis \
  --case-value MASLD --control-value Control

.venv/bin/python -m src.indonesia_extension compare \
  --otu OTU-table/indonesia_genus.tsv --metadata OTU-table/indonesia_metadata.csv \
  --table-format genus --sample-column sample_id --label-column diagnosis \
  --case-value MASLD --control-value Control \
  --split-column analysis_split --train-value train --test-value test --model SVM
```

Sebelum menjalankan `compare`, tetapkan pembagian latih/uji dan dokumentasikan kriterianya. Pelatihan Indonesia saja membutuhkan sedikitnya tiga kasus dan tiga kontrol pada set latih untuk validasi internal tiga lipatan; kedua kelas juga harus ada pada set uji. Satu hasil eksperimen tidak boleh dijalankan ulang ke nama keluaran yang sama secara diam-diam. Jika desain berubah, gunakan direktori keluaran baru dan catat alasannya.

## Batas resolusi taksonomi

Pipeline yang siap di sini bekerja pada genus, sesuai tabel enam kohor yang tersedia. Sepuluh takson tingkat spesies dan famili, heatmap spesies, serta skor LDA bergaya LEfSe dalam `Narration.md` belum dapat dihitung secara sah dari tabel genus. Keluaran tersebut memerlukan tabel pada resolusi yang sesuai, definisi analisis LDA/LEfSe, dan arti satuan “ELE” yang disepakati. Jangan menamai skor atau satuan lain sebagai LDA/LEfSe atau ELE hanya untuk melengkapi gambar.
