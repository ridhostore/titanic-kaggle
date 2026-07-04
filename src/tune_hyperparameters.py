"""
==============================================================================
Module: tune_hyperparameters.py
Deskripsi: Bayesian Optimization menggunakan Optuna untuk mencari 
           hyperparameter terbaik bagi XGBoost dan LightGBM.
Instruksi: Jalankan sekali via terminal: `python -m src.tune_hyperparameters`
==============================================================================
"""
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import feature engineering dari module kita
from .feature_engineering import engineer_features

def prepare_data_for_tuning():
    """Mempersiapkan data mentah menjadi fitur matang untuk tuning."""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    y = train_fe['Survived']
    X = train_fe.drop(cols_to_drop + ['Survived'], axis=1)
    X_test = test_fe.drop(cols_to_drop, axis=1)
    
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
    num_cols = ['Age', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'AgeMissing']
    
    # Imputation
    knn = KNNImputer(n_neighbors=5, weights='distance')
    X[num_cols] = knn.fit_transform(X[num_cols])
    X_test[num_cols] = knn.transform(X_test[num_cols])
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    X_test['Embarked'] = X_test['Embarked'].fillna(X_test['Embarked'].mode()[0])
    
    # Encoding & Scaling
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y

# ==============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# ==============================================================================
def objective_xgb(trial, X, y):
    """Objective function untuk XGBoost."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1
    }
    model = XGBClassifier(**param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_lgbm(trial, X, y):
    """Objective function untuk LightGBM."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    model = LGBMClassifier(**param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# ==============================================================================
# MAIN EXECUTION FOR TUNING
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Memulai Optuna Bayesian Optimization...")
    print("📂 Mempersiapkan data...")
    X, y = prepare_data_for_tuning()
    
    # --- Tune XGBoost ---
    print("\n🧠 [1/2] Optimizing XGBoost (100 trials)...")
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=100, show_progress_bar=True)
    
    print("\n✅ XGBoost Best Accuracy:", study_xgb.best_value)
    print("🏆 XGBoost Best Params:", study_xgb.best_params)
    
    # --- Tune LightGBM ---
    print("\n🧠 [2/2] Optimizing LightGBM (100 trials)...")
    study_lgbm = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=100, show_progress_bar=True)
    
    print("\n✅ LightGBM Best Accuracy:", study_lgbm.best_value)
    print("🏆 LightGBM Best Params:", study_lgbm.best_params)
    
    print("\n" + "="*60)
    print("🎯 TUNING SELESAI! Salin parameter di atas ke ensemble_modeling.py")
    print("="*60)