# Analisis Hilir Lintas Kohor

`run_analysis.py` mengolah tabel kelimpahan hasil pemrosesan 16S rRNA menjadi fitur genus terharmonisasi, lalu menjalankan analisis statistik dan evaluasi pembelajaran mesin lintas kohor.

## Asal dan batas data

Tabel yang diharapkan dalam folder lokal `OTU-table/` adalah keluaran antara yang dihasilkan oleh penelitian ini, bukan tabel eksternal siap pakai. Pembacaan mentah dan metadata berasal dari NCBI SRA untuk BioProject PRJNA1089365, PRJNA1152299, PRJNA1370285, PRJNA720274, PRJNA851946, dan PRJNA962007. Pemrosesan hulu menggunakan QIIME 2, DADA2, dan SILVA sebelum agregasi tingkat genus.

Folder `OTU-table/` tidak disertakan dalam Git. Daftar berkas masukan, negara, dan aturan label untuk setiap kohor didefinisikan pada `COHORTS` di `run_analysis.py`. Perintah, artefak, versi, dan parameter lengkap untuk tahap QIIME 2/DADA2/SILVA belum disertakan; skrip ini tidak mereproduksi tahap tersebut.

## Menjalankan

Setelah menyiapkan berkas masukan lokal dengan nama yang sesuai, jalankan dari akar proyek:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python src/run_analysis.py
```

Untuk membuat ulang gambar tanpa melatih ulang model, gunakan `--figures-only`. Mode ini tetap membaca `OTU-table/`, antara lain untuk menghitung ordinasi Bray-Curtis. Keluaran numerik disimpan di `results/`, sedangkan gambar analitik berada di `results/figures/`.

## Validasi dan keluaran

Pipeline menjalankan harmonisasi genus, transformasi *centered log-ratio*, ordinasi Bray-Curtis, PERMANOVA tersesuaikan kohor, meta-analisis efek acak, analisis sensitivitas, serta evaluasi SVM, random forest, MLP, dan CNN 1D. Pada validasi *nested leave-one-cohort-out*, seluruh penyaringan fitur, transformasi, reduksi dimensi, pemilihan model, dan ambang klasifikasi dipelajari tanpa kohor uji luar. Pembagian acak lima lipatan dilaporkan sebagai pembanding untuk menilai optimisme validasi, bukan sebagai estimasi kinerja utama.

Naskah dan berkas sumber ilustrasi khusus naskah disimpan secara lokal di `paper/`, yang diabaikan oleh Git. Tanpa salinan ilustrasi di folder tersebut, skrip melewati pembuatan ikhtisar naskah tetapi tetap menghasilkan gambar analitik lain. Salinan aset final untuk peninjauan tersedia di `assets/`.
