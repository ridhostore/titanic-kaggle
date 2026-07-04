"""
Module: feature_engineering.py
Deskripsi: Berisi logic untuk transformasi data mentah menjadi fitur prediktif.
"""
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan transformasi fitur tingkat lanjut berdasarkan domain knowledge.
    """
    df = df.copy()
    
    # A. Title Extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    
    # B. Family Dynamics
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # C. Deck Level
    df['Deck'] = df['Cabin'].str[0].fillna('Missing')
    deck_mapping = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE', 'F': 'FG', 'G': 'FG', 'Missing': 'Missing'}
    df['Deck'] = df['Deck'].map(deck_mapping)
    
    # D. Fare Normalization
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # E. Missingness Flag
    df['AgeMissing'] = df['Age'].isnull().astype(int)
    
    return df