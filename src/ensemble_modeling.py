"""
==============================================================================
Module: ensemble_modeling.py
Deskripsi: Pipeline preprocessing akhir, training Stacking Meta-Learner, 
           Cross-Validation scoring, dan prediksi. Menggunakan Optuna-tuned params.
==============================================================================
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .feature_engineering import engineer_features 

# Matikan warning yang tidak perlu untuk output terminal yang bersih
warnings.filterwarnings('ignore')

def preprocess_and_train(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Melakukan preprocessing, Cross-Validation, training Stacking Model, dan return prediksi.
    Returns: (submission_ids, predictions, cv_accuracy_score)
    """
    # 1. Feature Engineering
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)
    
    submission_ids = test_fe['PassengerId']
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    
    y = train_fe['Survived']
    X = train_fe.drop(cols_to_drop + ['Survived'], axis=1)
    X_test = test_fe.drop(cols_to_drop, axis=1)
    
    # 2. Imputation & Encoding
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
    num_cols = ['Age', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'AgeMissing']
    
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    X[num_cols] = knn_imputer.fit_transform(X[num_cols])
    X_test[num_cols] = knn_imputer.transform(X_test[num_cols])
    
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    X_test['Embarked'] = X_test['Embarked'].fillna(X_test['Embarked'].mode()[0])
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # ==========================================================================
    # 3. STACKING ENSEMBLE ARCHITECTURE (THE META-LEARNER)
    # ==========================================================================
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    
    xgb = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.04, 
        subsample=0.85, colsample_bytree=0.8, gamma=1.5, 
        min_child_weight=3, random_state=42, verbosity=0, n_jobs=-1
    )
    
    lgbm = LGBMClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.04, 
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, 
        reg_lambda=1.5, random_state=42, verbose=-1, n_jobs=-1
    )
    
    svc = SVC(
        probability=True, kernel='rbf', C=1.2, gamma='scale', random_state=42
    )
    
    meta_learner = LogisticRegression(
        C=0.1, max_iter=1000, random_state=42, n_jobs=-1
    )
    
    elite_stacking = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('svc', svc)],
        final_estimator=meta_learner,
        cv=5,
        passthrough=False,
        n_jobs=-1
    )
    
    # ==========================================================================
    # 4. CROSS-VALIDATION SCORING (ESTIMASI AKURASI MODEL)
    # ==========================================================================
    # Kita lakukan CV untuk mendapatkan estimasi akurasi model di data yang belum dilihat.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(elite_stacking, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_cv_score = cv_scores.mean()
    
    # ==========================================================================
    # 5. FINAL TRAINING & PREDICTION
    # ==========================================================================
    # Latih model pada SELURUH data training untuk prediksi akhir
    elite_stacking.fit(X, y)
    predictions = elite_stacking.predict(X_test)
    
    # Return 3 nilai: IDs, Prediksi, dan Skor CV
    return submission_ids, predictions, mean_cv_score