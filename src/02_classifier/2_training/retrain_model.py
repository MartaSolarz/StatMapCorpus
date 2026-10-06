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
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, precision_score,
    recall_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(" XGBoost not available - skipping XGBoost models")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print(" imblearn not available - skipping SMOTE variants")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ITERATIONS_DIR = BASE_DIR / "iterations"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = DATA_DIR / "train.parquet"
TEST_FILE = DATA_DIR / "test.parquet"
EMBEDDINGS_FILE = DATA_DIR / "sample" / "sample_200k_with_embeddings.parquet"


def load_train_data():
    print("Loading train.parquet...")
    df = pd.read_parquet(TRAIN_FILE)
    print(f"   Samples: {len(df)}")
    print(f"   YES: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    return df


def load_test_data():
    print("\nLoading test.parquet (FIXED test set)...")
    df = pd.read_parquet(TEST_FILE)
    print(f"   Samples: {len(df)}")
    print(f"   YES: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")
    print(f"    The test set is IDENTICAL to the baseline, so the comparison is fair")
    return df


def load_iteration_annotations(iteration_num):
    print(f"\nLoading iteration_{iteration_num} annotations...")

    iteration_dir = ITERATIONS_DIR / f"iteration_{iteration_num}"

    possible_files = [
        iteration_dir / "annotated.parquet",
        iteration_dir / "annotated_partial.parquet",
        iteration_dir / "annotations_progress.jsonl"
    ]

    df = None
    for file_path in possible_files:
        if file_path.exists():
            print(f"   Found: {file_path.name}")

            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.jsonl':
                records = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
                df = pd.DataFrame(records)

            break

    if df is None:
        raise FileNotFoundError(f"No annotation file found in {iteration_dir}")

    df = df[df['label'].isin(['YES', 'NO'])].copy()

    df['label'] = df['label'].map({'YES': 1, 'NO': 0})

    print(f"   Valid annotations: {len(df)}")
    print(f"   YES: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.2%})")
    print(f"   NO: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.2%})")

    return df


def merge_with_embeddings(df):
    print(f"\nMerging with embeddings...")

    print(f"   Loading embeddings...")
    df_embeddings = pd.read_parquet(EMBEDDINGS_FILE)

    df_merged = df.merge(df_embeddings, on='uid', how='inner')

    print(f"   After merge: {len(df_merged)} samples")

    missing = len(df) - len(df_merged)
    if missing > 0:
        print(f"    No embeddings for {missing} samples")

    return df_merged


def combine_datasets(baseline_df, iteration_df):
    print(f"\nCombining datasets...")

    baseline_uids = set(baseline_df['uid'])
    iteration_uids = set(iteration_df['uid'])

    duplicates = baseline_uids & iteration_uids
    if duplicates:
        print(f"    Found {len(duplicates)} duplicates; dropping them from the iteration")
        iteration_df = iteration_df[~iteration_df['uid'].isin(duplicates)].copy()

    combined_df = pd.concat([baseline_df, iteration_df], ignore_index=True)

    print(f"   Baseline: {len(baseline_df)} samples")
    print(f"   Iteration: {len(iteration_df)} samples")
    print(f"   Combined: {len(combined_df)} samples")
    print(f"   YES: {(combined_df['label'] == 1).sum()} ({(combined_df['label'] == 1).mean():.2%})")
    print(f"   NO: {(combined_df['label'] == 0).sum()} ({(combined_df['label'] == 0).mean():.2%})")

    return combined_df


def prepare_data(df):
    X = np.stack(df['l14_img'].values)
    y = df['label'].values
    uids = df['uid'].values

    return X, y, uids


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
    if not SMOTE_AVAILABLE:
        print("    SMOTE is not available; returning the original data")
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=min(3, (y_train == 1).sum() - 1)
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def train_and_evaluate_model(name, model_config, X_train, y_train, X_test, y_test, use_smote=False):
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    if use_smote:
        print(f"   + SMOTE oversampling")
    print(f"{'='*60}")

    scaler = None
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    if model_config['scale']:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)

    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train_processed, y_train)
    else:
        y_train_processed = y_train

    model = model_config['model']
    if 'XGBoost' in name:
        scale_pos_weight = (y_train_processed == 0).sum() / (y_train_processed == 1).sum()
        model.set_params(scale_pos_weight=scale_pos_weight)

    print("   Training...")
    model.fit(X_train_processed, y_train_processed)

    y_test_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_test_pred = model.predict(X_test_processed)

    results = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_test_pred_proba)
    results['optimal_threshold'] = optimal_threshold
    results['optimal_f1'] = optimal_f1

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
    print(f"   Accuracy:  {results['accuracy']:.4f}")
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
        print(f"   F1-score: {results['optimal_f1']:.4f}")


def analyze_validation_stability(y_val, y_pred_proba, model_name, n_bootstrap=1000):
    print(f"\nBootstrap stability analysis ({model_name}, n={n_bootstrap}):")

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_val), len(y_val), replace=True)
        y_boot = y_val[indices]
        y_pred_boot = (y_pred_proba[indices] >= 0.5).astype(int)

        f1_scores.append(f1_score(y_boot, y_pred_boot, zero_division=0))
        precision_scores.append(precision_score(y_boot, y_pred_boot, zero_division=0))
        recall_scores.append(recall_score(y_boot, y_pred_boot, zero_division=0))

    f1_scores = np.array(f1_scores)
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)

    print(f"   F1:        {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"              95% CI: [{np.percentile(f1_scores, 2.5):.4f}, {np.percentile(f1_scores, 97.5):.4f}]")
    print(f"   Precision: {precision_scores.mean():.4f} ± {precision_scores.std():.4f}")
    print(f"   Recall:    {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")

    if f1_scores.std() > 0.1:
        print(f"    NOTE: high instability (std > 0.1)")

    return {
        'f1_mean': float(f1_scores.mean()),
        'f1_std': float(f1_scores.std()),
        'f1_ci_lower': float(np.percentile(f1_scores, 2.5)),
        'f1_ci_upper': float(np.percentile(f1_scores, 97.5))
    }


def track_all_models_over_iterations(iteration_num):
    print("\n" + "="*80)
    print("TRENDS FOR EVERY MODEL ACROSS ITERATIONS")
    print("="*80)

    baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
    if not baseline_files:
        print(" No baseline results found")
        return None

    with open(baseline_files[-1], 'r') as f:
        baseline_data = json.load(f)

    history = {}

    for model_data in baseline_data['all_models']:
        model_name = model_data['Model']
        history[model_name] = [{
            'iteration': 0,
            'f1': model_data['F1'],
            'recall': model_data['Recall'],
            'precision': model_data['Precision'],
            'pr_auc': model_data['PR-AUC']
        }]

    for i in range(1, iteration_num + 1):
        files = list(RESULTS_DIR.glob(f"iteration_{i}_results_*.json"))
        if files:
            with open(files[-1], 'r') as f:
                data = json.load(f)

            for model_data in data['all_models']:
                model_name = model_data['Model']
                if model_name not in history:
                    history[model_name] = []

                history[model_name].append({
                    'iteration': i,
                    'f1': model_data['F1'],
                    'recall': model_data['Recall'],
                    'precision': model_data['Precision'],
                    'pr_auc': model_data['PR-AUC']
                })

    print(f"\nPER-MODEL comparison (baseline → iteration_{iteration_num}):\n")
    print(f"{'Model':<30} {'Baseline':<12} {'Iter_{}':<12} {'Change':<15} {'Trend'}".format(iteration_num))
    print("-" * 85)

    trends = {}
    for model_name, iterations in sorted(history.items()):
        if len(iterations) >= 2:
            baseline_f1 = iterations[0]['f1']
            latest_f1 = iterations[-1]['f1']
            change = latest_f1 - baseline_f1
            change_pct = (change / baseline_f1 * 100) if baseline_f1 > 0 else 0

            arrow = "" if change > 0 else "" if change < 0 else ""
            print(f"{model_name:<30} {baseline_f1:<12.4f} {latest_f1:<12.4f} {change:+.4f} ({change_pct:+5.1f}%) {arrow}")

            trends[model_name] = {
                'baseline_f1': baseline_f1,
                'latest_f1': latest_f1,
                'change': change
            }

    print(f"\nTOP 3 - largest improvement:")
    sorted_trends = sorted(trends.items(), key=lambda x: x[1]['change'], reverse=True)
    for model_name, trend in sorted_trends[:3]:
        print(f"   {model_name}: {trend['change']:+.4f}")

    return trends


def compare_same_models_baseline_vs_iteration(all_results, iteration_num):
    print("\n" + "="*80)
    print("DETAILED COMPARISON: the same model, baseline vs iteration")
    print("="*80)

    baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
    if not baseline_files:
        print(" No baseline results found")
        return None

    with open(baseline_files[-1], 'r') as f:
        baseline_data = json.load(f)

    baseline_map = {}
    for model_data in baseline_data['all_models']:
        baseline_map[model_data['Model']] = model_data

    print(f"\n{'Model':<35} {'Baseline F1':<15} {'Iteration F1':<15} {'Change'}")
    print("-" * 80)

    comparisons = []
    for result in all_results:
        model_name = result['name'] + (' + SMOTE' if result['use_smote'] else '')

        if model_name in baseline_map:
            baseline_f1 = baseline_map[model_name]['F1']
            iter_f1 = result['results']['f1']
            change = iter_f1 - baseline_f1
            change_pct = (change / baseline_f1 * 100) if baseline_f1 > 0 else 0

            arrow = "" if change > 0 else "" if change < 0 else ""
            print(f"{model_name:<35} {baseline_f1:<15.4f} {iter_f1:<15.4f} {arrow} {change:+.4f} ({change_pct:+.1f}%)")

            comparisons.append({
                'model': model_name,
                'baseline_f1': baseline_f1,
                'iteration_f1': iter_f1,
                'change': change
            })

    return comparisons


def compare_with_baseline(new_results, iteration_num):
    print("\n" + "="*80)
    print("COMPARISON AGAINST BASELINE (on the same validation set)")
    print("="*80)

    baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
    if not baseline_files:
        print(" No baseline results to compare against")
        return

    with open(baseline_files[-1], 'r') as f:
        baseline_data = json.load(f)

    baseline_metrics = baseline_data['best_model']['metrics']

    best_new = max(new_results, key=lambda x: x['results']['f1'])
    new_metrics = best_new['results']

    print(f"\n{'Metric':<15} {'Baseline':<12} {'Iteration_{iteration_num}':<15} {'Change':<15}")
    print("-" * 60)

    metrics_to_compare = ['f1', 'precision', 'recall', 'roc_auc', 'pr_auc']

    improvements = []
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics[metric]
        new_val = new_metrics[metric]
        change = new_val - baseline_val
        change_pct = (change / baseline_val * 100) if baseline_val > 0 else 0

        arrow = "" if change > 0 else "" if change < 0 else ""

        print(f"{metric.upper():<15} {baseline_val:<12.4f} {new_val:<15.4f} {arrow} {change:+.4f} ({change_pct:+.1f}%)")

        if change > 0:
            improvements.append(metric)

    print("\n" + "="*80)
    print("INTERPRETATION (the validation set is IDENTICAL to the baseline):")
    print("="*80)

    f1_improved = new_metrics['f1'] > baseline_metrics['f1']
    recall_improved = new_metrics['recall'] > baseline_metrics['recall']
    pr_auc_improved = new_metrics['pr_auc'] > baseline_metrics['pr_auc']

    if f1_improved and recall_improved:
        print("THE MODEL IMPROVED SUBSTANTIALLY")
        print(f"   F1 rose: {baseline_metrics['f1']:.4f} → {new_metrics['f1']:.4f}")
        print(f"   Recall rose: {baseline_metrics['recall']:.4f} → {new_metrics['recall']:.4f}")
        print(f"\n   RECOMMENDATION: use this model and run iteration_{iteration_num + 1}")
    elif recall_improved:
        print("THE MODEL IMPROVED")
        print(f"   Recall rose: {baseline_metrics['recall']:.4f} → {new_metrics['recall']:.4f}")
        print(f"   → The model finds MORE maps")
        if not f1_improved:
            print(f"    F1 fell slightly; that is the trade-off, more recall at the cost of precision")
        print(f"\n   RECOMMENDATION: a good improvement, run iteration_{iteration_num + 1}")
    elif pr_auc_improved:
        print("THE MODEL IMPROVED SLIGHTLY")
        print(f"   PR-AUC rose: {baseline_metrics['pr_auc']:.4f} → {new_metrics['pr_auc']:.4f}")
        print(f"   → Overall prediction quality is better for imbalanced data")
        print(f"\n   RECOMMENDATION: run iteration_{iteration_num + 1}; the next round may give a larger gain")
    elif len(improvements) > 0:
        print(f"PARTIAL IMPROVEMENT")
        print(f"   Metrics that improved: {', '.join(improvements)}")
        print(f"\n   RECOMMENDATION: run iteration_{iteration_num + 1}")
    else:
        print(" The model did not improve")
        print(f"\n   Possible reasons:")
        print(f"   1. Too little new data ({iteration_num * 600} samples)")
        print(f"   2. The new samples were too similar to the baseline")
        print(f"   3. The model needs more iterations (try iteration_{iteration_num + 1})")
        print(f"   4. This model may have reached a plateau")
        print(f"\n   RECOMMENDATION: try iteration_{iteration_num + 1}, or consider a different approach")

    return best_new


def save_model(best_result, iteration_num, X_train, y_train):
    print(f"\nSaving the iteration_{iteration_num} model...")

    name = best_result['name']
    use_smote = best_result['use_smote']
    model = best_result['model']
    scaler = best_result['scaler']

    X_train_processed = X_train.copy()
    if scaler:
        X_train_processed = scaler.fit_transform(X_train)

    if use_smote:
        X_train_processed, y_train_processed = apply_smote(X_train_processed, y_train)
    else:
        y_train_processed = y_train

    if 'XGBoost' in name:
        scale_pos_weight = (y_train_processed == 0).sum() / (y_train_processed == 1).sum()
        model.set_params(scale_pos_weight=scale_pos_weight)

    model.fit(X_train_processed, y_train_processed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = MODELS_DIR / f"model_iteration_{iteration_num}_{timestamp}.pkl"

    model_package = {
        'model': model,
        'scaler': scaler,
        'name': name,
        'use_smote': use_smote,
        'results': best_result['results'],
        'timestamp': timestamp,
        'iteration': iteration_num,
        'optimal_threshold': best_result['results']['optimal_threshold']
    }

    joblib.dump(model_package, model_filename)
    print(f"   Model saved: {model_filename}")

    latest_link = MODELS_DIR / "model_latest.pkl"
    joblib.dump(model_package, latest_link)
    print(f"   Latest link: {latest_link}")

    return model_filename


def save_results(all_results, best_result, iteration_num, baseline_comparison=None):
    print(f"\nSaving the iteration_{iteration_num} results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"iteration_{iteration_num}_results_{timestamp}.json"

    results_data = {
        'iteration': iteration_num,
        'timestamp': timestamp,
        'best_model': {
            'name': best_result['name'],
            'use_smote': best_result['use_smote'],
            'metrics': best_result['results']
        },
        'all_models': []
    }

    for result in all_results:
        metrics = result['results']
        bootstrap = metrics.get('bootstrap', {})

        model_data = {
            'Model': result['name'] + (' + SMOTE' if result['use_smote'] else ''),
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Accuracy': metrics.get('accuracy', metrics['f1']),
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Optimal F1': metrics.get('optimal_f1', metrics['f1']),
            'Bootstrap_CI': {
                'f1_mean': bootstrap.get('f1_mean'),
                'f1_std': bootstrap.get('f1_std'),
                'f1_ci_lower': bootstrap.get('f1_ci_lower'),
                'f1_ci_upper': bootstrap.get('f1_ci_upper')
            } if bootstrap else None
        }
        results_data['all_models'].append(model_data)

    if baseline_comparison:
        results_data['baseline_comparison'] = baseline_comparison

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"   Results saved: {results_file}")
    print(f"   Saved {len(all_results)} models")

    return results_file

def main(iteration_num=1):
    print("\n" + "="*80)
    print(f"MODEL RETRAIN - ITERATION {iteration_num}")
    print("="*80)

    train_df = load_train_data()

    test_df = load_test_data()

    iteration_df = load_iteration_annotations(iteration_num)

    iteration_df = merge_with_embeddings(iteration_df)

    extended_train_df = combine_datasets(train_df, iteration_df)

    X_train, y_train, uid_train = prepare_data(extended_train_df)
    X_test, y_test, uid_test = prepare_data(test_df)

    print(f"\nData summary:")
    print(f"   Train (extended): {len(y_train)} samples")
    print(f"      YES: {(y_train == 1).sum()} ({(y_train == 1).mean():.2%})")
    print(f"      NO: {(y_train == 0).sum()} ({(y_train == 0).mean():.2%})")
    print(f"   Test (FIXED):     {len(y_test)} samples")
    print(f"      YES: {(y_test == 1).sum()} ({(y_test == 1).mean():.2%})")
    print(f"      NO: {(y_test == 0).sum()} ({(y_test == 0).mean():.2%})")
    print(f"   The test set is IDENTICAL to the baseline")

    models = get_models()
    all_results = []

    for name, model_config in models.items():
        result = train_and_evaluate_model(
            name, model_config, X_train, y_train, X_test, y_test, use_smote=False
        )
        all_results.append(result)

        if SMOTE_AVAILABLE and name != 'SVM_RBF':
            result_smote = train_and_evaluate_model(
                name, model_config, X_train, y_train, X_test, y_test, use_smote=True
            )
            all_results.append(result_smote)

    comparisons = compare_same_models_baseline_vs_iteration(all_results, iteration_num)

    trends = track_all_models_over_iterations(iteration_num)

    best_result = compare_with_baseline(all_results, iteration_num)

    if best_result:
        baseline_files = sorted(RESULTS_DIR.glob("baseline_results_*.json"))
        baseline_comparison = None
        if baseline_files:
            with open(baseline_files[-1], 'r') as f:
                baseline_data = json.load(f)
            baseline_metrics = baseline_data['best_model']['metrics']

            baseline_comparison = {
                'baseline_f1': baseline_metrics['f1'],
                'baseline_recall': baseline_metrics['recall'],
                'baseline_precision': baseline_metrics['precision'],
                'iteration_f1': best_result['results']['f1'],
                'iteration_recall': best_result['results']['recall'],
                'iteration_precision': best_result['results']['precision'],
                'f1_change': best_result['results']['f1'] - baseline_metrics['f1'],
                'recall_change': best_result['results']['recall'] - baseline_metrics['recall'],
                'precision_change': best_result['results']['precision'] - baseline_metrics['precision']
            }

        results_file = save_results(all_results, best_result, iteration_num, baseline_comparison)

    if best_result:
        model_filename = save_model(best_result, iteration_num, X_train, y_train)

    print("\n" + "="*80)
    print(f"RETRAIN COMPLETE - ITERATION {iteration_num}")
    print("="*80)

    print(f"\nNext steps:")
    print(f"   1. Check whether the model improved (above)")
    print(f"   2. If it did: run iteration_{iteration_num + 1}")
    print(f"      python3 active_learning.py --iteration {iteration_num + 1}")
    print(f"   3. If it did not: work out why, and consider a different approach")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Refit the model after an active-learning iteration')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number to retrain from (default: 1)')

    args = parser.parse_args()

    main(iteration_num=args.iteration)
