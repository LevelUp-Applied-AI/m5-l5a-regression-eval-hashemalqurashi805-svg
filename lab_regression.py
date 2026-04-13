"""
Module 5 Week A — Lab: Regression & Evaluation
Author: Hashem Al-Qurashi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, r2_score, confusion_matrix)

# المهمة 1: تحميل البيانات
def load_data(filepath="data/telecom_churn.csv"):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# المهمة 2: تقسيم البيانات مع التقسيم الطبقي (Stratification)
def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # نستخدم stratify فقط إذا كان الهدف تصنيفياً (مثل Churned)
    stratify_val = y if y.dtype in ['int64', 'object'] else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_val)

# المهمة 3: خط أنابيب التصنيف (Logistic Regression)
def build_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

# المهمة 4: خط أنابيب Ridge
def build_ridge_pipeline(alpha=1.0):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alpha))
    ])

# المهمة 5: خط أنابيب Lasso
def build_lasso_pipeline(alpha=0.1):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=alpha))
    ])

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        # الميزات الرقمية المختارة
        numeric_features = ["tenure", "monthly_charges", "total_charges", "num_support_calls", 
                            "senior_citizen", "has_partner", "has_dependents"]

        # --- الجزء الأول: التصنيف (التنبؤ بترك العميل - Churn) ---
        print("\n" + "="*50)
        print("PART 1: CLASSIFICATION (CHURN)")
        print("="*50)
        
        df_cls = df[numeric_features + ["churned"]].dropna()
        X_train, X_test, y_train, y_test = split_data(df_cls, "churned")

        pipe_cls = build_logistic_pipeline()
        pipe_cls.fit(X_train, y_train)
        y_pred = pipe_cls.predict(X_test)

        print("\n1. Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # المهمة 6: التحقق المتقاطع (Cross Validation)
        print("\n2. Cross-Validation (Stratified 5-Fold):")
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe_cls, X_train, y_train, cv=cv_splitter, scoring="accuracy")
        print(f"CV Accuracy Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

        # --- الجزء الثاني: الانحدار (التنبؤ بالتكاليف الشهرية) ---
        print("\n" + "="*50)
        print("PART 2: REGRESSION (MONTHLY CHARGES)")
        print("="*50)
        
        reg_features = ["tenure", "total_charges", "num_support_calls", "senior_citizen", "has_partner", "has_dependents"]
        df_reg = df[reg_features + ["monthly_charges"]].dropna()
        X_tr, X_te, y_tr, y_te = split_data(df_reg, "monthly_charges")

        # تقييم Ridge (المهمة 4)
        ridge_pipe = build_ridge_pipeline()
        ridge_pipe.fit(X_tr, y_tr)
        print(f"Ridge R2 Score: {r2_score(y_te, ridge_pipe.predict(X_te)):.3f}")

        # تقييم Lasso (المهمة 5)
        lasso_pipe = build_lasso_pipeline()
        lasso_pipe.fit(X_tr, y_tr)
        print(f"Lasso R2 Score: {r2_score(y_te, lasso_pipe.predict(X_te)):.3f}")

        # طباعة المعاملات لمعرفة أهمية الميزات (Feature Importance)
        print("\nLasso Coefficients:")
        coeffs = dict(zip(reg_features, lasso_pipe.named_steps['regressor'].coef_))
        for feature, coef in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True):
            print(f"{feature}: {coef:.3f}")

# --- المهمة 7: ملخص وتحليل النتائج ---
"""
SUMMARY OF FINDINGS:
1. Feature Importance: 'total_charges' and 'tenure' appear to be the most significant predictors.
2. Recall vs. Accuracy: In churn prediction, Recall is prioritized because missing a potential 
   churner (False Negative) is more costly to the business than a False Positive.
3. Model Regularization: Lasso successfully highlighted key features by shrinking less 
   relevant coefficients, providing a more interpretable model than standard OLS.
"""