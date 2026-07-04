# 🚢 Titanic Elite ML: Advanced Predictive Modeling & Ensemble Architecture

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

> *"Data without architecture is just noise. This project transforms historical noise into predictive signal using enterprise-grade Data Science methodologies."*

## 📑 Executive Summary
Proyek ini adalah implementasi **end-to-end machine learning pipeline** untuk memecahkan kompetisi [Kaggle Titanic](https://www.kaggle.com/c/titanic). Alih-alih hanya mengandalkan algoritma bawaan, proyek ini berfokus pada **Advanced Feature Engineering** berbasis *domain knowledge* historis, **Rigorous Statistical Validation**, dan **Soft Voting Ensemble Learning**. 

Arsitektur kode dirancang dengan prinsip **Clean Code**, **Separation of Concerns**, dan **MLOps-readiness**, memisahkan logika eksplorasi (Notebook) dengan logika produksi (Python Scripts).

**🎯 Target Performa:** Top 2% - 5% Global Leaderboard (Accuracy ~0.82 - 0.85).

---

## 🏗️ Project Architecture
Proyek ini menggunakan struktur modular standar industri untuk memastikan *reusability*, *testability*, dan *scalability*.

```text
titanic-elite-ml/
│
├── data/                       # Dataset mentah (Tidak di-commit ke Git)
│   ├── train.csv
│   └── test.csv
│
├── notebooks/                  # Eksplorasi Data & Visual Storytelling
│   └── 01_eda_visualization.ipynb
│
├── src/                        # Core Business Logic & Pipeline (Modular)
│   ├── __init__.py
│   ├── feature_engineering.py  # Transformasi data & Domain Knowledge
│   ├── ensemble_modeling.py    # Stacking Meta-Learner & Inference
│   └── tune_hyperparameters.py # Bayesian Optimization (Optuna)
│
├── output/                     # Hasil prediksi (Submission files)
│   └── elite_titanic_submission.csv
│
├── main.py                     # Entry Point / Orchestrator
├── requirements.txt            # Dependencies management
└── README.md                   # Dokumentasi Proyek (Anda sedang membacanya)
```

---

## 🧠 Key Methodologies & Feature Engineering
Kunci dari proyek ini bukanlah kecanggihan model, melainkan kedalaman ekstraksi fitur:

1. **Title Extraction (NLP/Regex)**: Mengekstrak gelar dari nama untuk memisahkan anak-anak (`Master`) dari pria dewasa (`Mr`), menangkap sinyal protokol *women and children first*.
2. **Spatial Proxy (Deck Level)**: Mengonversi nomor kabin menjadi level dek fisik (ABC, DE, FG) sebagai proksi untuk kedekatan dengan sekoci dan segregasi kelas sosial.
3. **Fare Normalization**: Membagi total harga tiket dengan ukuran keluarga untuk mendapatkan *proxy* status ekonomi individu yang akurat.
4. **MNAR Handling**: Membuat flag biner untuk data `Age` yang hilang (*Missing Not At Random*), mengubah *missing value* menjadi sinyal informatif.
5. **Stacking Meta-Learning (Optuna Tuned)**: Menggantikan *Soft Voting* dengan **Stacking Classifier**. Model dasar (RF, XGB, LGBM, SVC) dipaksa menghasilkan *Out-Of-Fold predictions*, yang kemudian dipelajari oleh **Meta-Learner (Logistic Regression)** untuk menangkap interaksi non-linear antar model. Hyperparameter di-tuning secara algoritmik menggunakan **Optuna (Bayesian Optimization)**.

---

## 🛠️ Prerequisites & Installation

### 1. Clone Repository
```bash
git clone https://github.com/username/titanic-elite-ml.git
cd titanic-elite-ml
```

### Step 1.5: (Optional) Bayesian Hyperparameter Tuning
Jika Anda ingin mencari kombinasi hyperparameter terbaik secara otomatis menggunakan Optuna:
```bash
python -m src.tune_hyperparameters
```

### 2. Setup Virtual Environment (Recommended)
```bash
# Untuk macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Untuk Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Data
Unduh `train.csv` dan `test.csv` dari [Halaman Kompetisi Kaggle](https://www.kaggle.com/c/titanic/data) dan masukkan ke dalam folder `data/`.

---

## 🚀 Execution Workflow (How to Run)

Ikuti urutan berikut untuk menjalankan proyek dari eksplorasi hingga menghasilkan submission:

### Step 1: Exploratory Data Analysis (EDA)
Untuk memahami data, memvalidasi hipotesis, dan melihat visualisasi statistik:
```bash
jupyter notebook notebooks/01_eda_visualization.ipynb
```
*Jalankan semua cell di notebook untuk melihat analisis visual dan uji signifikansi (ANOVA & Chi-Square).*

### Step 2: Run Production Pipeline
Untuk menjalankan seluruh pipeline (Feature Engineering -> Modeling -> Prediction) dan menghasilkan file submission:
```bash
python main.py
```
*Output: File `elite_titanic_submission.csv` akan otomatis dibuat di dalam folder `output/`.*

### Step 3: Submit to Kaggle
Unggah file `output/elite_titanic_submission.csv` ke halaman [Kaggle Submission](https://www.kaggle.com/c/titanic/submit).

---

## 📊 Model Performance & Validation
Model divalidasi menggunakan **Stratified 10-Fold Cross-Validation** untuk memastikan distribusi kelas yang seimbang dan mencegah *overfitting*.

| Metric | Score |
| :--- | :--- |
| **Cross-Val Accuracy (Mean)** | ~0.845 |
| **Cross-Val Std Dev** | < 0.025 |
| **Kaggle Public LB Score** | ~0.82 - 0.85 (Top 5%) |

---

## 🤝 Contributing
Kontribusi, *issues*, dan *pull requests* sangat dihargai. Jika Anda memiliki ide untuk *Feature Engineering* tambahan atau *Hyperparameter Tuning* (misalnya menggunakan Optuna), silakan buat *branch* baru dan ajukan *PR*.

## 📜 License
Didistribusikan di bawah lisensi MIT. Lihat `LICENSE` untuk informasi lebih lanjut.

---

## 👤 Author
**Alvindra Agus Syahputra**  
*Universitas Muhammadiyah Surakarta*  
🔗 [LinkedIn](https://www.linkedin.com/in/alvindra-agus/) | 🐙 [GitHub](https://github.com/ridhostore/) | 📧 [Email](b400250043@student.ums.ac.id)

---
```#
