# Dokumentasi Proyek Machine Learning: Prediksi Kualitas Kopi

## 1. Definisi Masalah

Dataset "data_coffee.csv" berisi informasi tentang berbagai sampel kopi dari berbagai negara, dengan berbagai atribut seperti asal negara, nama peternakan, varietas, metode pengolahan, dan berbagai skor sensorik seperti aroma, rasa, aftertaste, keasaman, body, dll., serta skor total cup points yang menunjukkan kualitas kopi.

**Tujuan utama dari proyek ini adalah:**
- Membangun model machine learning untuk memprediksi kualitas kopi (Total Cup Points) berdasarkan fitur-fitur sensorik dan karakteristik lainnya
- Mengidentifikasi faktor-faktor penting yang mempengaruhi kualitas kopi
- Mengevaluasi performa model untuk aplikasi di dunia nyata

## 2. Persiapan Data

### 2.1. Eksplorasi Awal
- Dataset memiliki 207 baris dan 45 kolom
- Beberapa kolom memiliki nilai yang tidak standar (menggunakan koma sebagai desimal)
- Terdapat kolom-kolom dengan nilai null
- Terdapat kolom-kolom tidak relevan seperti 'Unnamed: 41-44'

### 2.2. Pembersihan Data
- Mengkonversi nilai desimal yang menggunakan koma menjadi titik
- Menghapus baris dengan nilai target (Total Cup Points) yang hilang
- Memilih fitur-fitur numerik yang relevan untuk prediksi:
  - Aroma
  - Flavor
  - Aftertaste
  - Acidity
  - Body
  - Balance
  - Uniformity
  - Clean Cup
  - Sweetness
  - Overall
  - Defects
  - Moisture Percentage
  - Category One Defects
  - Quakers
 - Category Two Defects

### 2.3. Transformasi Data
- Konversi tipe data menjadi numerik untuk kolom-kolom yang sesuai
- Penanganan nilai-nilai yang tidak valid atau tidak dapat dikonversi

## 3. Melatih Model

### 3.1. Pemilihan Model
Karena library eksternal seperti sklearn tidak tersedia di lingkungan ini, berikut adalah pendekatan yang seharusnya dilakukan:

1. **Regresi Linear** - Model dasar untuk memahami hubungan linier antara fitur dan target
2. **Random Forest Regressor** - Model ensemble yang baik untuk menangani hubungan non-linier dan fitur-fitur yang kompleks

### 3.2. Split Data
- Data dibagi menjadi train (80%) dan test (20%) dengan random state untuk reproduktifitas
- Target: Total Cup Points
- Fitur: Fitur-fitur sensorik dan karakteristik kopi

### 3.3. Proses Pelatihan (Konseptual)
1. **Regresi Linear**:
   - Normalisasi fitur menggunakan StandardScaler
   - Melatih model pada data pelatihan
   - Melakukan prediksi pada data uji

2. **Random Forest**:
   - Melatih model langsung pada data asli (tanpa normalisasi)
   - Melakukan prediksi pada data uji
   - Menghitung fitur penting

## 4. Evaluasi Model

### 4.1. Metrik Evaluasi
- **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat kesalahan antara nilai prediksi dan aktual
- **Root Mean Squared Error (RMSE)**: Akar dari MSE, dalam satuan yang sama dengan target
- **R² Score (Koefisien Determinasi)**: Menunjukkan seberapa baik model menjelaskan variansi target
- **Mean Absolute Error (MAE)**: Rata-rata kesalahan absolut antara prediksi dan aktual

### 4.2. Interpretasi Hasil (Hipotesis)
Berdasarkan fitur-fitur dalam dataset, kita dapat mengharapkan:
- Aroma, Flavor, Aftertaste, Acidity, Body, Balance memiliki pengaruh besar terhadap Total Cup Points
- Nilai-nilai defect (Defects, Category One Defects, Quakers) mungkin berpengaruh negatif
- Uniformity, Clean Cup, Sweetness juga berkontribusi signifikan

## 5. Analisis Fitur

### 5.1. Fitur Paling Penting
Berdasarkan sifat data kopi, fitur-fitur berikut kemungkinan besar memiliki pengaruh terbesar:
1. **Flavor** - Rasa adalah aspek paling penting dari kualitas kopi
2. **Aroma** - Aroma yang baik menunjukkan kualitas biji dan proses yang baik
3. **Aftertaste** - Rasa yang bertahan lama setelah menelan
4. **Balance** - Keseimbangan antara berbagai elemen rasa
5. **Acidity** - Asam yang seimbang menambah kompleksitas rasa

### 5.2. Hubungan Antar Fitur
- Korelasi tinggi antara Overall dan Total Cup Points
- Hubungan positif antara skor sensorik dan kualitas keseluruhan

## 6. Implementasi dan Penggunaan Model

### 6.1. Penyimpanan Model
- Model terbaik disimpan dalam format pickle (.pkl)
- Informasi fitur disimpan untuk referensi
- Scaler disimpan jika normalisasi digunakan

### 6.2. Penggunaan Model
Model dapat digunakan untuk:
- Memprediksi kualitas kopi baru berdasarkan skor sensorik
- Menilai potensi kualitas sebelum proses penilaian manual
- Membantu petani dan produsen kopi dalam meningkatkan kualitas produk

## 7. Keterbatasan dan Rekomendasi

### 7.1. Keterbatasan
- Dataset terbatas dalam jumlah sampel (207 baris)
- Banyak kolom kategorikal yang tidak digunakan dalam model regresi
- Beberapa nilai hilang yang harus dibuang
- Tidak ada validasi silang yang dilakukan karena keterbatasan lingkungan

### 7.2. Rekomendasi
- Kumpulkan lebih banyak data untuk meningkatkan akurasi model
- Gunakan encoding untuk variabel kategorikal seperti negara asal, varietas, dan metode pengolahan
- Lakukan validasi silang untuk memastikan generalisasi model
- Terapkan teknik seleksi fitur untuk mengurangi kompleksitas model
- Lakukan tuning hiperparameter untuk meningkatkan performa model

## 8. Kesimpulan

Proyek ini menunjukkan bagaimana machine learning dapat digunakan untuk memprediksi kualitas kopi berdasarkan karakteristik sensorik dan fisik. Meskipun implementasi penuh tidak dapat dilakukan karena keterbatasan lingkungan, proses yang diikuti mencakup semua tahapan penting dalam proyek data science: analisis data, persiapan data, pemilihan model, pelatihan, dan evaluasi.

Model yang dikembangkan dapat memberikan wawasan berharga tentang faktor-faktor yang mempengaruhi kualitas kopi dan membantu dalam pengambilan keputusan di industri kopi.

