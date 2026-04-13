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
    return pd.read_csv(filepath)

# المهمة 2: تقسيم البيانات (مهم جداً للاختبار التلقائي)
def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # التقسيم الطبقي للحفاظ على التوزيع (التحدي)
    stratify_val = y if y.dtype in ['int64', 'object'] else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_val)

# المهمة 3: خط أنابيب التصنيف مع أوزان متوازنة (تمديد التحدي)
def build_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

# دالة التقييم (Task 3)
def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# المهمة 6: التحقق المتقاطع الطبقي (تمديد التحدي)
def run_cross_validation(pipeline, X, y):
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # نستخدم f1 كمعيار تقييم أدق للبيانات غير المتوازنة
    return cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='f1')

# المهمة 4: خط أنابيب Ridge
def build_ridge_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])

# المهمة 5: خط أنابيب Lasso (تمديد التحدي)
def build_lasso_pipeline(alpha=0.1):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=alpha))
    ])

# دالة تقييم الانحدار (Task 4 & 5)
def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        # الميزات الرقمية
        features = ["tenure", "monthly_charges", "total_charges", "num_support_calls", 
                    "senior_citizen", "has_partner", "has_dependents"]

        # --- تصنيف ---
        X_train, X_test, y_train, y_test = split_data(df[features + ["churned"]].dropna(), "churned")
        pipe_cls = build_logistic_pipeline()
        print(f"Classification Metrics: {evaluate_classifier(pipe_cls, X_train, X_test, y_train, y_test)}")
        
        # --- انحدار ---
        reg_features = ["tenure", "total_charges", "num_support_calls", "senior_citizen", "has_partner", "has_dependents"]
        X_tr, X_te, y_tr, y_te = split_data(df[reg_features + ["monthly_charges"]].dropna(), "monthly_charges")
        
        ridge_pipe = build_ridge_pipeline()
        print(f"Ridge Metrics: {evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)}")
        
        lasso_pipe = build_lasso_pipeline()
        print(f"Lasso Metrics: {evaluate_regressor(lasso_pipe, X_tr, X_te, y_tr, y_te)}")

# المهمة 7: التحليل النهائي
"""
1. تم استخدام StandardScaler لضمان استقرار النماذج الخطية.
2. تم استخدام class_weight='balanced' للتعامل مع عدم توازن البيانات في الـ Churn.
3. Lasso أظهر قدرة على تقليص المعاملات غير المهمة (Feature Selection).
"""