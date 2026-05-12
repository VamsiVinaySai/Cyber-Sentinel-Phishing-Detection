import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ONLY pure URL/lexical features — nothing that requires visiting the page.
# This MUST match the return order of extract_features() in features.py.
CSV_FEATURE_COLS = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
    'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb',
    'URLCharProb', 'NoOfSubDomain', 'HasObfuscation',
    'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL',
    'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
]

def train():
    print("Loading dataset...")
    df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
    df = df.sample(n=min(50000, len(df)), random_state=42)

    available = [c for c in CSV_FEATURE_COLS if c in df.columns]
    missing   = [c for c in CSV_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING - columns missing from CSV: {missing}")

    X = df[available].fillna(0).values
    y = df['label'].values  # 1 = Legit, 0 = Phishing

    joblib.dump(available, 'feature_cols.pkl')
    print(f"Training on {len(available)} URL-only features: {available}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, 'phishing_model.pkl')
    print("Saved: phishing_model.pkl + feature_cols.pkl")

if __name__ == "__main__":
    train()


# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import joblib

# # MUST match the return order of extract_features() in features.py
# CSV_FEATURE_COLS = [
#     'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
#     'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb',
#     'URLCharProb', 'NoOfSubDomain', 'HasObfuscation',
#     'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL',
#     'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
#     'NoOfEquals', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
#     'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
#     'LongestPathTokenLength', 'URLTitleMatchScore', 'HasSocialNet',
#     'DomainTitleMatchScore', 'URLDomainRatio', 'CharRepeatRate',
#     'NoOfSlashInURL',
# ]

# def train():
#     print("Loading PhiUSIIL dataset...")
#     df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
#     df = df.sample(n=min(50000, len(df)), random_state=42)

#     available = [c for c in CSV_FEATURE_COLS if c in df.columns]
#     missing   = [c for c in CSV_FEATURE_COLS if c not in df.columns]
#     print(f"Using {len(available)} columns. Missing from CSV: {missing}")

#     X = df[available].fillna(0).values
#     y = df['label'].values  # 1=Legit, 0=Phishing

#     # Save so app.py uses the same columns in the same order
#     joblib.dump(available, 'feature_cols.pkl')

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     print("Training model...")
#     clf = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=20,
#         min_samples_leaf=2,
#         class_weight='balanced',
#         n_jobs=-1,
#         random_state=42,
#     )
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
#     print(classification_report(y_test, y_pred))

#     joblib.dump(clf, 'phishing_model.pkl')
#     print("Saved: phishing_model.pkl + feature_cols.pkl")

# if __name__ == "__main__":
#     train()
