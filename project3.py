# 1. Required Libraries Import Karein
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Reproducibility ke liye random seed set karein
RANDOM_STATE = 42

print("Libraries imported successfully!")

# ==========================================
# 2. Dataset Load & Simulation (Demo Purpose)
# ==========================================
# Note: Real scenario me aap Kaggle ka Credit Card Fraud dataset (CSV file) load karenge:
# df = pd.read_csv('creditcard.csv')

# Yahan hum ek dummy imbalanced dataset simulate kar rahe hain demonstration ke liye:
from sklearn.datasets import make_classification
X_dummy, y_dummy = make_classification(
    n_samples=100000, n_features=30, n_informative=25, 
    n_classes=2, weights=[0.995, 0.005], random_state=RANDOM_STATE
)

# DataFrame convert kar rahe hain
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
df = pd.DataFrame(X_dummy, columns=feature_cols)
df['Class'] = y_dummy

print(f"Dataset Shape: {df.shape}")
print("Class Distribution:\n", df['Class'].value_counts())

# ==========================================
# 3. Data Preprocessing & Feature Scaling
# ==========================================
# 'Time' aur 'Amount' columns ko scale karna zaroori hai kyunki baaki features pehle se PCA scaled hain
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

# Purane unscaled columns drop karein
df = df.drop(['Time', 'Amount'], axis=1)

# Features (X) aur Target (y) ko alag karein
X = df.drop('Class', axis=1)
y = df['Class']

# ==========================================
# 4. Train-Test Split (Stratified)
# ==========================================
# Imbalanced data me stratified split karna zaroori hai taaki train aur test dono me fraud cases maintain rahein
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

# ==========================================
# 5. Handling Imbalance using SMOTE
# ==========================================
# ⚠️ IMPORTANT: SMOTE hamesha sirf TRAINING data par lagana chahiye, TEST data par nahi!
print("Applying SMOTE on training data...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE - Fraud cases: {sum(y_train == 1)}, Normal cases: {sum(y_train == 0)}")
print(f"After SMOTE  - Fraud cases: {sum(y_train_smote == 1)}, Normal cases: {sum(y_train_smote == 0)}")

# ==========================================
# 6. Model Training (XGBoost)
# ==========================================
print("Training XGBoost Classifier...")
# Imbalanced data ke liye scale_pos_weight bhi use kar sakte hain, par SMOTE ke baad standard XGBoost acha kaam karta hai
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)

xgb_model.fit(X_train_smote, y_train_smote)
print("Model training completed!")

# ==========================================
# 7. Model Evaluation & Performance Metrics
# ==========================================
# Test data par predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC-AUC & PR-AUC scores
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"Precision-Recall AUC Score: {pr_auc:.4f}")