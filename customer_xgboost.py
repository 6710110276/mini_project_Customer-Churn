import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier

def load_and_preprocess_raw_data(file_path: str):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


def build_xgboost_pipeline(X: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )

    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.8,
        scale_pos_weight=1.8,
        random_state=42,
        eval_metric=['logloss', 'auc'],
        early_stopping_rounds=30
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])

    return model_pipeline


def print_metrics(set_name: str, model: Pipeline, X_data: pd.DataFrame, y_data: pd.Series, threshold: float = 0.47):
    y_proba = model.predict_proba(X_data)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_data, y_pred)
    prec = precision_score(y_data, y_pred, zero_division=0)
    rec = recall_score(y_data, y_pred, zero_division=0)
    f1 = f1_score(y_data, y_pred, zero_division=0)
    auc = roc_auc_score(y_data, y_proba)

    print(f"=== {set_name} (Threshold: {threshold}) ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}\n")


def evaluate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.47):
    print(" XGBoost Customer Churn Evaluation \n")
    
    print_metrics("Validation Set", model, X_val, y_val, threshold)
    print_metrics("Test Set", model, X_test, y_test, threshold)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    print("[ Detailed Classification Report (Test Set) ]")
    print(classification_report(y_test, y_test_pred, target_names=['No Churn (0)', 'Churn (1)'], zero_division=0))


def predict_custom_sample(model: Pipeline, input_data: dict, threshold: float = 0.47):
    df_single = pd.DataFrame([input_data])
    churn_proba = model.predict_proba(df_single)[0, 1]
    is_churn = churn_proba >= threshold

    print("="*50 + "\n")
    print(" ผลการทำนายข้อมูลลูกค้า ")
    
    if is_churn:
        confidence = churn_proba * 100
        print(f"ผลการทำนาย : [ Risk ] มีแนวโน้มย้ายค่าย (Churn)")
        print(f"ความมั่นใจ  : {confidence:.2f}% (โอกาสย้ายค่าย)")
    else:
        confidence = (1 - churn_proba) * 100
        print(f"ผลการทำนาย : [ Safe ] มีแนวโน้มใช้งานต่อ (No Churn)")
        print(f"ความมั่นใจ  : {confidence:.2f}% (โอกาสใช้งานต่อ)")
        
    print("="*50 + "\n")


def prompt_user_input(X_reference: pd.DataFrame) -> dict:
    print("   กรอกข้อมูลลูกค้าครบทั้ง 19 ฟีเจอร์เพื่อทำนายผล   ")
    print("   (กด Enter เพื่อใช้ค่า Default ในแต่ละช่องได้)    ")

    default_values = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 85.0
    }

    user_input = {}

    for col in X_reference.columns:
        default_val = default_values.get(col, '')

        if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            val_str = input(f"ป้อนค่า {col} (ค่าเริ่มต้น: {default_val}): ").strip()
            if val_str == "":
                user_input[col] = default_val
            else:
                user_input[col] = float(val_str) if '.' in val_str else int(val_str)

        elif col == 'SeniorCitizen':
            labels = ['No', 'Yes']
            options_prompt = " [" + ", ".join([f"'{lbl}'({idx+1})" for idx, lbl in enumerate(labels)]) + "]"
            
            default_idx = 1 if default_val == 0 else 2
            val_str = input(f"ป้อน {col}{options_prompt} (ค่าเริ่มต้น: {default_idx} - {labels[default_idx-1]}): ").strip()

            if val_str == "":
                user_input[col] = default_val
            elif val_str in ['1', '2']:
                user_input[col] = int(val_str) - 1 
            else:
                print(f"  [!] ไม่พบตัวเลือกหมายเลข '{val_str}' ระบบจะใช้ค่าเริ่มต้น: {default_val}")
                user_input[col] = default_val

        else:
            unique_options = sorted([str(opt) for opt in X_reference[col].unique()])
            options_prompt_list = [f"'{opt}'({idx+1})" for idx, opt in enumerate(unique_options)]
            options_prompt = " [" + ", ".join(options_prompt_list) + "]"
            
            default_idx = 1
            for idx, opt in enumerate(unique_options):
                if opt.lower() == str(default_val).lower():
                    default_idx = idx + 1
                    break

            val_str = input(f"ป้อน {col}{options_prompt} (ค่าเริ่มต้น: {default_idx} - {unique_options[default_idx-1]}): ").strip()

            if val_str == "":
                user_input[col] = unique_options[default_idx - 1]
            elif val_str.isdigit() and 1 <= int(val_str) <= len(unique_options):
                user_input[col] = unique_options[int(val_str) - 1]
            else:
                print(f"  [!] ไม่พบตัวเลือกหมายเลข '{val_str}' ระบบจะใช้ค่าเริ่มต้น: {unique_options[default_idx-1]}")
                user_input[col] = unique_options[default_idx - 1]

    if user_input.get('PhoneService') == 'No':
        user_input['MultipleLines'] = 'No phone service'

    if user_input.get('InternetService') == 'No':
        for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            user_input[service] = 'No internet service'

    return user_input


def main():
    data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    print("กำลังโหลดและเตรียมข้อมูล")
    X, y = load_and_preprocess_raw_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print("กำลังสร้างและแปลงข้อมูลด้วย Pipeline")
    pipeline = build_xgboost_pipeline(X_tr, y_tr)

    preprocessor = pipeline.named_steps['preprocessor']
    X_tr_transformed = preprocessor.fit_transform(X_tr, y_tr)
    X_val_eval = preprocessor.transform(X_val)
    
    pipeline.fit(
        X_tr, y_tr,
        classifier__eval_set=[(X_val_eval, y_val)],
        classifier__verbose=False
    )

    evaluate_model(pipeline, X_val, y_val, X_test, y_test, threshold=0.47)

    while True:
        choice = input("\nต้องการทดสอบป้อนข้อมูลลูกค้าเพื่อทำนายผลหรือไม่? (y/n): ").strip().lower()
        if choice == 'y':
            print("="*50 + "\n")
            custom_customer = prompt_user_input(X)
            predict_custom_sample(pipeline, custom_customer, threshold=0.47)
        else:
            print("\nจบการทำงาน")
            break


if __name__ == '__main__':
    main()