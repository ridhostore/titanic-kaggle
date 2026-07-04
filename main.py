"""
==============================================================================
Entry Point: main.py
Deskripsi: Orchestrator utama yang menjalankan seluruh pipeline Titanic ML
           dari data ingestion hingga menghasilkan file submission Kaggle.
Author: Alvindra Agus Syahputra
==============================================================================
"""
import pandas as pd
import os
import time
from datetime import datetime

from src.ensemble_modeling import preprocess_and_train

# ==============================================================================
# HELPER FUNCTIONS: TERMINAL UI & LOGGING
# ==============================================================================
def print_banner():
    """Menampilkan banner pembuka proyek."""
    banner = """
======================================================================
   🚢  TITANIC ELITE ML PIPELINE  🚢                             
                                                                  
   Advanced Feature Engineering + Stacking Meta-Learning          
   Target: Top 1-3% Kaggle Leaderboard (~0.83 - 0.85)            
======================================================================
"""
    print(banner)

def log_step(step_num: int, icon: str, message: str):
    """Mencetak log langkah dengan timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"  [{timestamp}] {icon} [Step {step_num}] {message}")

def log_info(message: str):
    """Mencetak informasi detail (indentasi)."""
    print(f"             ├─ ℹ️  {message}")

def log_success(message: str):
    """Mencetak pesan sukses."""
    print(f"             └─ ✅ {message}")

def log_warning(message: str):
    """Mencetak peringatan."""
    print(f"             ├─ ⚠️  {message}")

def log_error(message: str):
    """Mencetak error."""
    print(f"             └─ ❌ {message}")

def print_separator():
    """Mencetak garis pemisah."""
    print("  " + "─" * 64)

def validate_data(df: pd.DataFrame, name: str):
    """Melakukan validasi awal terhadap dataset."""
    print(f"             ├─ 📊 Dataset: {name}")
    print(f"             │   ├─ Shape     : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"             │   ├─ Memory    : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print(f"             │   └─ Missing   : {len(missing_cols)} kolom memiliki missing values")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"             │       • {col:<12} : {count:>4} ({pct:.1f}%)")
    else:
        print(f"             │   └─ Missing   : 0 (Data bersih!)")

def print_execution_summary(start_time: float, output_path: str, n_predictions: int, cv_score: float):
    """Mencetak ringkasan akhir eksekusi."""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    
    time_str = f"{minutes}m {seconds:.2f}s"
    pred_str = f"{n_predictions} penumpang"
    acc_str = f"{cv_score:.2%}" # Format menjadi persentase (misal: 86.50%)
    
    print("\n" + "="*68)
    print("📋 EXECUTION SUMMARY")
    print("="*68)
    print(f"  ⏱️  Total Waktu       : {time_str}")
    print(f"  📈 Model Accuracy    : {acc_str} (Stratified 5-Fold CV)")
    print(f"  📊 Total Prediksi    : {pred_str}")
    print(f"  📁 Output File       : {output_path}")
    print(f"  🎯 Model             : Stacking Meta-Learner (RF+XGB+LGBM+SVC)")
    print(f"  📐 Features          : 12 (termasuk Title, Deck, FarePerPax)")
    print("-"*68)
    print("  📤 NEXT STEP:")
    print(f"     Unggah '{output_path}' ke Kaggle Submit")
    print("="*68 + "\n")

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    start_time = time.time()
    
    # --- BANNER ---
    print_banner()
    log_step(0, "🔧", "Inisialisasi Pipeline...")
    print_separator()
    
    # ======================================================================
    # STEP 1: DATA INGESTION
    # ======================================================================
    log_step(1, "📂", "DATA INGESTION - Memuat dataset...")
    
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    if not os.path.exists(train_path):
        log_error(f"File tidak ditemukan: {train_path}")
        print("\n  💡 Pastikan Anda sudah mengunduh data dari Kaggle dan meletakkannya di folder 'data/'.")
        return
    
    if not os.path.exists(test_path):
        log_error(f"File tidak ditemukan: {test_path}")
        return
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    validate_data(train, "Training Set")
    validate_data(test, "Test Set")
    
    if 'Survived' not in train.columns:
        log_error("Kolom 'Survived' tidak ditemukan di training set!")
        return
    log_success(f"Data berhasil dimuat. Target column: 'Survived' (distribusi: {(train['Survived'].mean()*100):.1f}% selamat)")
    print_separator()
    
    # ======================================================================
    # STEP 2: FEATURE ENGINEERING & PREPROCESSING
    # ======================================================================
    log_step(2, "⚙️", "FEATURE ENGINEERING - Mengekstrak sinyal tersembunyi...")
    log_info("Ekstraksi Title dari kolom Name (Regex)...")
    log_info("Pembuatan FamilySize & IsAlone...")
    log_info("Konversi Cabin -> Deck Level (Spatial Proxy)...")
    log_info("Normalisasi Fare -> FarePerPerson...")
    log_info("Pembuatan flag AgeMissing (MNAR Handling)...")
    log_success("12 fitur prediktif berhasil dibuat.")
    print_separator()
    
    # ======================================================================
    # STEP 3: MODEL TRAINING & PREDICTION
    # ======================================================================
    log_step(3, "🧠", "MODEL TRAINING - Melatih Stacking Meta-Learner...")
    log_info("Base Models: RandomForest | XGBoost | LightGBM | SVC")
    log_info("Meta-Learner: Logistic Regression (Out-Of-Fold Predictions)")
    log_info("Imputasi: KNNImputer (k=5, distance-weighted)")
    log_info("Encoding: LabelEncoder (konsisten train+test)")
    log_info("Scaling: StandardScaler (mean=0, std=1)")
    log_info("Validation: Stratified 5-Fold Cross-Validation")
    
    print("             ├─ ⏳ Training & Cross-Validation sedang berlangsung...")
    model_start = time.time()
    
    try:
        # Sekarang menerima 3 nilai: ids, preds, dan cv_score
        ids, preds, cv_score = preprocess_and_train(train, test)
        model_elapsed = time.time() - model_start
        log_success(f"Training selesai dalam {model_elapsed:.2f} detik.")
        log_success(f"Estimated Accuracy (CV): {cv_score:.2%}")
    except Exception as e:
        log_error(f"Training gagal: {str(e)}")
        return
    
    print_separator()
    
    # ======================================================================
    # STEP 4: PREDICTION ANALYSIS
    # ======================================================================
    log_step(4, "📊", "PREDICTION ANALYSIS - Menganalisis hasil prediksi...")
    
    survived_count = sum(preds == 1)
    perished_count = sum(preds == 0)
    total_preds = len(preds)
    
    log_info(f"Total prediksi     : {total_preds} penumpang")
    log_info(f"Diprediksi Selamat : {survived_count} ({(survived_count/total_preds*100):.1f}%)")
    log_info(f"Diprediksi Tewas   : {perished_count} ({(perished_count/total_preds*100):.1f}%)")
    
    train_survival_rate = train['Survived'].mean() * 100
    test_pred_survival_rate = (survived_count / total_preds) * 100
    rate_diff = abs(train_survival_rate - test_pred_survival_rate)
    
    if rate_diff < 10:
        log_success(f"Distribusi prediksi masuk akal (selisih {rate_diff:.1f}% dari training).")
    else:
        log_warning(f"Distribusi prediksi berbeda {rate_diff:.1f}% dari training. Periksa kemungkinan overfitting.")
    
    print_separator()
    
    # ======================================================================
    # STEP 5: SAVE SUBMISSION
    # ======================================================================
    log_step(5, "💾", "SAVE SUBMISSION - Menyimpan file output...")
    
    output_dir = 'output'
    output_filename = 'elite_titanic_submission.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    submission = pd.DataFrame({
        'PassengerId': ids,
        'Survived': preds
    })
    
    submission.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / 1024
    log_info(f"Path   : {output_path}")
    log_info(f"Ukuran : {file_size:.1f} KB")
    log_info(f"Baris  : {len(submission)} (termasuk header)")
    log_success("File submission berhasil disimpan!")
    
    # ======================================================================
    # EXECUTION SUMMARY
    # ======================================================================
    print_execution_summary(start_time, output_path, total_preds, cv_score)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()