import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    f1_score, precision_score, recall_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(" XGBoost is not installed. Install it with: pip install xgboost")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "train.parquet"
TEST_FILE = DATA_DIR / "test.parquet"


def setup_directories():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def load_train_data():
    print("Loading training data (train.parquet)...")
    df = pd.read_parquet(TRAIN_FILE)

    print(f"   Samples: {len(df)}")
    print(f"   YES (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    print(f"   Imbalance ratio: {(df['label'] == 0).sum() / (df['label'] == 1).sum():.1f}:1")

    X = np.stack(df['l14_img'].values)
    y = df['label'].values
    uids = df['uid'].values

    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    return X, y, uids, df


def load_test_data():
    print("\nLoading test data (test.parquet, FIXED)...")
    df = pd.read_parquet(TEST_FILE)

    print(f"   Samples: {len(df)}")
    print(f"   YES (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")

    X = np.stack(df['l14_img'].values)
    y = df['label'].values
    uids = df['uid'].values

    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    return X, y, uids, df


def get_models():
    models = {}

    models['LogisticRegression'] = {
        'model': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'scale': True,
        'description': 'Logistic Regression with class_weight=balanced'
    }

    models['LogisticRegression_L1'] = {
        'model': LogisticRegression(
            class_weight='balanced',
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        ),
        'scale': True,
        'description': 'Logistic Regression L1 (sparse)'
    }

    models['RandomForest'] = {
        'model': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'scale': False,
        'description': 'Random Forest with class_weight=balanced'
    }

    models['RandomForest_Subsample'] = {
        'model': RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced_subsample',
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'scale': False,
        'description': 'Random Forest with balanced_subsample'
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = {
            'model': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'scale': False,
            'description': 'XGBoost with scale_pos_weight'
        }

    models['SVM_RBF'] = {
        'model': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'scale': True,
        'description': 'SVM with an RBF kernel'
    }

    return models


def apply_smote(X_train, y_train, sampling_strategy=0.3):
    print(f"\nApplying SMOTE (sampling_strategy={sampling_strategy})...")
    print(f"   Before SMOTE:")
    print(f"      Samples: {len(y_train)}")
    print(f"      YES: {(y_train == 1).sum()}")
    print(f"      NO: {(y_train == 0).sum()}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=3
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE:")
    print(f"      Samples: {len(y_resampled)}")
    print(f"      YES: {(y_resampled == 1).sum()} ({(y_resampled == 1).mean():.2%})")
    print(f"      NO: {(y_resampled == 0).sum()} ({(y_resampled == 0).mean():.2%})")

    return X_resampled, y_resampled


def train_and_evaluate_model(name, model_config, X_train, y_train, X_test, y_test, use_smote=False):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"   {model_config['description']}")
    if use_smote:
        print(f"   + SMOTE oversampling")
    print(f"{'='*60}")

    scaler = None
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    if model_config['scale']:
        print("   Scaling features...")
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)

    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train_processed, y_train, sampling_strategy=0.3)
    else:
        y_train_processed = y_train

    model = model_config['model']
    if 'XGBoost' in name:
        scale_pos_weight = (y_train_processed == 0).sum() / (y_train_processed == 1).sum()
        model.set_params(scale_pos_weight=scale_pos_weight)
        print(f"   XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    print("   Training...")
    model.fit(X_train_processed, y_train_processed)

    y_test_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_test_pred = model.predict(X_test_processed)

    results = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_test_pred_proba)
    results['optimal_threshold'] = optimal_threshold
    results['optimal_f1'] = optimal_f1

    y_test_pred_optimal = (y_test_pred_proba >= optimal_threshold).astype(int)
    results['metrics_optimal_threshold'] = calculate_metrics(y_test, y_test_pred_optimal, y_test_pred_proba)

    print_results(results, optimal_threshold)

    return {
        'model': model,
        'scaler': scaler,
        'results': results,
        'name': name,
        'use_smote': use_smote
    }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'confusion_matrix': {
            'TP': int(tp), 'FP': int(fp),
            'TN': int(tn), 'FN': int(fn)
        }
    }


def find_optimal_threshold(y_true, y_pred_proba):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]

    return optimal_threshold, optimal_f1


def print_results(results, optimal_threshold=None):
    print(f"\nResults (threshold=0.5):")
    print(f"   Accuracy:  {results['accuracy']:.4f}  (misleading under imbalance)")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-score:  {results['f1']:.4f}")
    print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"   PR-AUC:    {results['pr_auc']:.4f}")

    cm = results['confusion_matrix']
    print(f"\n   Confusion Matrix:")
    print(f"      TP: {cm['TP']:4d}  FP: {cm['FP']:4d}")
    print(f"      FN: {cm['FN']:4d}  TN: {cm['TN']:4d}")

    if optimal_threshold:
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"   F1 at the optimal threshold: {results['optimal_f1']:.4f}")


def compare_models(all_results):
    print("\n" + "="*80)
    print("COMPARISON OF EVERY MODEL")
    print("="*80)

    comparison = []
    for result in all_results:
        metrics = result['results']
        comparison.append({
            'Model': result['name'] + (' + SMOTE' if result['use_smote'] else ''),
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Optimal F1': metrics['optimal_f1']
        })

    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('F1', ascending=False)

    print("\nRanked by F1 (threshold=0.5):")
    print(df_comparison.to_string(index=False))

    best_idx = df_comparison['F1'].idxmax()
    best_model_name = df_comparison.loc[best_idx, 'Model']

    print(f"\nBEST MODEL: {best_model_name}")
    print(f"   F1-score: {df_comparison.loc[best_idx, 'F1']:.4f}")
    print(f"   Precision: {df_comparison.loc[best_idx, 'Precision']:.4f}")
    print(f"   Recall: {df_comparison.loc[best_idx, 'Recall']:.4f}")
    print(f"   PR-AUC: {df_comparison.loc[best_idx, 'PR-AUC']:.4f}")

    best_result = None
    for result in all_results:
        model_name = result['name'] + (' + SMOTE' if result['use_smote'] else '')
        if model_name == best_model_name:
            best_result = result
            break

    return best_result, df_comparison


def save_model(best_result):
    print(f"\nSaving the best model...")

    name = best_result['name']
    use_smote = best_result['use_smote']
    model = best_result['model']
    scaler = best_result['scaler']

    print(f"   The model is already fitted on the whole training set")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = MODELS_DIR / f"baseline_model_{name}_{timestamp}.pkl"

    model_package = {
        'model': model,
        'scaler': scaler,
        'name': name,
        'use_smote': use_smote,
        'results': best_result['results'],
        'timestamp': timestamp,
        'optimal_threshold': best_result['results']['optimal_threshold']
    }

    joblib.dump(model_package, model_filename)
    print(f"   Model saved: {model_filename}")

    latest_link = MODELS_DIR / "baseline_model_latest.pkl"
    joblib.dump(model_package, latest_link)
    print(f"   Latest link: {latest_link}")

    return model_filename


def save_results(df_comparison, best_result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"baseline_results_{timestamp}.json"

    results = {
        'timestamp': timestamp,
        'best_model': {
            'name': best_result['name'],
            'use_smote': best_result['use_smote'],
            'metrics': best_result['results']
        },
        'all_models': df_comparison.to_dict(orient='records')
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written: {results_file}")


def main():
    print("\n" + "="*80)
    print("BASELINE MODEL - TRAINING")
    print("="*80)

    setup_directories()

    X_train, y_train, uid_train, df_train = load_train_data()
    X_test, y_test, uid_test, df_test = load_test_data()

    print("\nData summary:")
    print(f"   Train: {len(y_train)} samples")
    print(f"   Test:  {len(y_test)} samples (FIXED across every iteration)")

    models = get_models()
    print(f"\nModels to try: {len(models)}")

    all_results = []

    for name, model_config in models.items():
        result = train_and_evaluate_model(
            name, model_config, X_train, y_train, X_test, y_test, use_smote=False
        )
        all_results.append(result)

        if name not in ['SVM_RBF']:
            result_smote = train_and_evaluate_model(
                name, model_config, X_train, y_train, X_test, y_test, use_smote=True
            )
            all_results.append(result_smote)

    best_result, df_comparison = compare_models(all_results)

    model_filename = save_model(best_result)

    save_results(df_comparison, best_result)

    print("\n" + "="*80)
    print("BASELINE MODEL - DONE")
    print("="*80)
    print(f"\nModel saved: {model_filename}")
    print(f"Test set: 801 samples (FIXED across every iteration)")
    print(f"Next step: active_learning.py")


if __name__ == "__main__":
    main()
