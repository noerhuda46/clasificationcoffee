import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset yang telah diproses
df = pd.read_csv('processed_coffee_data.csv', sep=';')

# Identifikasi kolom numerik yang relevan untuk prediksi
numeric_features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 
                   'Uniformity', 'Clean Cup', 'Sweetness', 'Overall', 'Defects', 
                   'Moisture Percentage', 'Category One Defects', 'Quakers', 'Category Two Defects']

# Hanya gunakan kolom numerik yang tersedia di dataset
available_features = [col for col in numeric_features if col in df.columns]

# Target: Total Cup Points
target_col = 'Total Cup Points'

# Hapus baris dengan nilai target yang hilang
df = df.dropna(subset=[target_col])

# Hapus baris dengan nilai fitur yang hilang (untuk kemudahan)
df_model = df[available_features + [target_col]].dropna()

print(f"Jumlah sampel untuk pelatihan: {len(df_model)}")
print(f"Fitur yang digunakan: {available_features}")

# Pisahkan fitur dan target
X = df_model[available_features]
y = df_model[target_col]

print(f"Dimensi X: {X.shape}")
print(f"Dimensi y: {y.shape}")

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dimensi X_train: {X_train.shape}")
print(f"Dimensi X_test: {X_test.shape}")

# Normalisasi fitur (opsional, tergantung model)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Model Regresi Linear
print("\n--- Regresi Linear ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluasi
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"Mean Squared Error: {mse_lr:.4f}")
print(f"Root Mean Squared Error: {rmse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print(f"Mean Absolute Error: {mae_lr:.4f}")

# 2. Model Random Forest
print("\n--- Random Forest Regressor ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Gunakan data yang tidak dinormalisasi untuk Random Forest

# Prediksi
y_pred_rf = rf_model.predict(X_test)

# Evaluasi
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Mean Squared Error: {mse_rf:.4f}")
print(f"Root Mean Squared Error: {rmse_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")
print(f"Mean Absolute Error: {mae_rf:.4f}")

# Menampilkan fitur penting dari Random Forest
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFitur paling penting (Random Forest):")
print(feature_importance)

# Visualisasi prediksi vs aktual untuk model terbaik
best_model_name = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
best_predictions = y_pred_rf if r2_rf > r2_lr else y_pred_lr

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Nilai Aktual')
plt.ylabel('Nilai Prediksi')
plt.title(f'Prediksi vs Aktual - {best_model_name}')
plt.show()

# Simpan model terbaik
import joblib

if r2_rf > r2_lr:
    joblib.dump(rf_model, 'best_coffee_model.pkl')
    joblib.dump(scaler, 'coffee_scaler.pkl')  # Simpan scaler meskipun tidak digunakan untuk RF
    print(f"\nModel Random Forest disimpan sebagai 'best_coffee_model.pkl'")
else:
    joblib.dump(lr_model, 'best_coffee_model.pkl')
    joblib.dump(scaler, 'coffee_scaler.pkl')
    print(f"\nModel Regresi Linear disimpan sebagai 'best_coffee_model.pkl'")

# Simpan juga informasi fitur
with open('coffee_features.txt', 'w') as f:
    f.write('\n'.join(available_features))
    print("Fitur-fitur yang digunakan disimpan dalam 'coffee_features.txt'")
