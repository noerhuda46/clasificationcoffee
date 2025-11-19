import pandas as pd
import numpy as np

# Membaca dataset
df = pd.read_csv('data_coffee.csv', sep=';')

print("Dimensi dataset:", df.shape)
print("\nNama kolom:")
print(df.columns.tolist())

print("\nInfo dataset:")
print(df.info())

print("\nStatistik deskriptif:")
print(df.describe())

print("\n5 baris pertama dataset:")
print(df.head())

print("\nJumlah nilai null per kolom:")
print(df.isnull().sum())

print("\nTipe data per kolom:")
print(df.dtypes)

# Melihat apakah ada duplikat
print(f"\nJumlah duplikat: {df.duplicated().sum()}")

# Kolom-kolom numerik yang akan digunakan untuk analisis
numeric_cols = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Overall', 'Defects', 'Total Cup Points', 'Moisture Percentage', 'Category One Defects', 'Quakers']

# Cek dan konversi kolom numerik yang benar-benar berisi data numerik
for col in numeric_cols:
    if col in df.columns:
        # Cek apakah kolom bisa dikonversi ke numerik
        temp_series = pd.to_numeric(df[col], errors='coerce')
        if not temp_series.isna().all():  # Jika tidak semua nilai NaN setelah konversi
            df[col] = temp_series

print("\nStatistik deskriptif setelah konversi tipe data:")
print(df[numeric_cols].describe())

# Melihat distribusi nilai-nilai unik untuk kolom kategorikal
categorical_cols = [col for col in df.columns if col not in numeric_cols]
print(f"\nJumlah nilai unik per kolom kategorikal (pertama 10):")
for col in categorical_cols[:10]:  # hanya menampilkan 10 kolom pertama
    print(f"{col}: {df[col].nunique()} unik")

# Menyimpan dataset yang telah diproses
df.to_csv('processed_coffee_data.csv', index=False, sep=';')
print(f"\nDataset yang telah diproses disimpan sebagai 'processed_coffee_data.csv' dengan dimensi {df.shape}")
