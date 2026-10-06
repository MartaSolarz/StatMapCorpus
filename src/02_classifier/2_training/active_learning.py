import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ITERATIONS_DIR = BASE_DIR / "iterations"

POOL_FILE = DATA_DIR / "pool.parquet"
TRAIN_FILE = DATA_DIR / "train.parquet"
MODEL_LATEST = MODELS_DIR / "model_latest.pkl"
BASELINE_MODEL = MODELS_DIR / "baseline_model_latest.pkl"


def load_model():
    print(" Loading model...")

    if MODEL_LATEST.exists():
        model_file = MODEL_LATEST
        print(f"    Found retrained model: {model_file.name}")
    elif BASELINE_MODEL.exists():
        model_file = BASELINE_MODEL
        print(f"    Using baseline model (no retrain available): {model_file.name}")
    else:
        raise FileNotFoundError(f"No model found. Checked: {MODEL_LATEST} and {BASELINE_MODEL}")

    model_package = joblib.load(model_file)

    print(f"   Model: {model_package['name']}")
    print(f"   SMOTE: {model_package['use_smote']}")
    print(f"   Timestamp: {model_package['timestamp']}")

    if 'iteration' in model_package:
        print(f"    Model from iteration: {model_package['iteration']}")
        print(f"   F1: {model_package['results']['f1']:.4f}")
        print(f"   Recall: {model_package['results']['recall']:.4f}")
    else:
        print(f"    Baseline model")
        print(f"   Baseline F1: {model_package['results']['f1']:.4f}")
        print(f"   Baseline Recall: {model_package['results']['recall']:.4f}")

    print(f"   Optimal threshold: {model_package['optimal_threshold']:.4f}")

    return model_package


def load_pool_data():
    print("\n Loading pool data...")

    df = pd.read_parquet(POOL_FILE)

    print(f"   Samples in pool: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    X = np.stack(df['l14_img'].values)
    print(f"   X shape: {X.shape}")

    return df, X


def predict_probabilities(model_package, X, batch_size=10000):
    print("\n Predicting probabilities...")

    model = model_package['model']
    scaler = model_package['scaler']

    if scaler:
        print("   Scaling features...")
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    n_samples = len(X_processed)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"   Samples: {n_samples:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Number of batches: {n_batches}")

    y_pred_proba = []

    for i in tqdm(range(n_batches), desc="   Predicting"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        batch_proba = model.predict_proba(X_processed[start_idx:end_idx])[:, 1]
        y_pred_proba.append(batch_proba)

    y_pred_proba = np.concatenate(y_pred_proba)

    print(f"\n    Prediction complete")
    print(f"   Probabilities: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

    return y_pred_proba


def calculate_uncertainty_entropy(y_pred_proba):
    epsilon = 1e-10
    p = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    return entropy


def calculate_uncertainty_margin(y_pred_proba):
    margin = 1 - np.abs(y_pred_proba - 0.5) * 2

    return margin


def calculate_uncertainty_least_confident(y_pred_proba):
    max_prob = np.maximum(y_pred_proba, 1 - y_pred_proba)

    uncertainty = 1 - max_prob

    return uncertainty


def select_samples_to_annotate(df, y_pred_proba, n_samples=600, strategy='entropy', invalid_rate=0.38):
    n_with_buffer = int(n_samples / (1 - invalid_rate))

    print(f"\n Selecting samples for annotation (PURE UNCERTAINTY)...")
    print(f"   NOTE: pure uncertainty may select edge cases only.")
    print(f"   Strategy: {strategy}")
    print(f"   Target valid annotations: {n_samples}")
    print(f"   Expected share of invalid URLs: {invalid_rate:.1%}")
    print(f"   Drawing with buffer: {n_with_buffer} samples")
    print(f"   (expecting ~{int(n_with_buffer * (1-invalid_rate))} valid)")

    if strategy == 'entropy':
        uncertainty = calculate_uncertainty_entropy(y_pred_proba)
    elif strategy == 'margin':
        uncertainty = calculate_uncertainty_margin(y_pred_proba)
    elif strategy == 'least_confident':
        uncertainty = calculate_uncertainty_least_confident(y_pred_proba)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df_with_uncertainty = df.copy()
    df_with_uncertainty['pred_proba'] = y_pred_proba
    df_with_uncertainty['uncertainty'] = uncertainty

    df_sorted = df_with_uncertainty.sort_values('uncertainty', ascending=False)

    df_selected = df_sorted.head(n_with_buffer).copy()

    print(f"\n    Statistics of the selected samples:")
    print(f"      Uncertainty: min={df_selected['uncertainty'].min():.4f}, max={df_selected['uncertainty'].max():.4f}, mean={df_selected['uncertainty'].mean():.4f}")
    print(f"      Pred proba: min={df_selected['pred_proba'].min():.4f}, max={df_selected['pred_proba'].max():.4f}, mean={df_selected['pred_proba'].mean():.4f}")

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(df_selected['pred_proba'], bins=bins)

    print(f"\n      pred_proba distribution (selected samples):")
    for i in range(len(bins) - 1):
        pct = hist[i] / len(df_selected) * 100
        print(f"         [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d} samples ({pct:5.1f}%)")

    print(f"\n    Statistics of the whole pool (for comparison):")
    print(f"      Uncertainty: mean={uncertainty.mean():.4f}, std={uncertainty.std():.4f}")
    print(f"      Pred proba: mean={y_pred_proba.mean():.4f}, std={y_pred_proba.std():.4f}")

    n_potential_positive = (df_selected['pred_proba'] > 0.5).sum()
    print(f"\n      Potential YES (proba > 0.5): {n_potential_positive} ({n_potential_positive/len(df_selected):.1%})")

    return df_selected, df_with_uncertainty


def select_samples_stratified(df, y_pred_proba, n_samples=600, strategy='entropy',
                               invalid_rate=0.38, positive_ratio=0.4):
    n_with_buffer = int(n_samples / (1 - invalid_rate))

    print(f"\n Selecting samples for annotation (STRATIFIED UNCERTAINTY)...")
    print(f"    Stratification prevents selecting edge cases only")
    print(f"   Strategy: {strategy} (stratified)")
    print(f"   Target valid annotations: {n_samples}")
    print(f"   Positive ratio: {positive_ratio:.1%}")
    print(f"   Expected share of invalid URLs: {invalid_rate:.1%}")
    print(f"   Drawing with buffer: {n_with_buffer} samples")

    if strategy == 'entropy':
        uncertainty = calculate_uncertainty_entropy(y_pred_proba)
    elif strategy == 'margin':
        uncertainty = calculate_uncertainty_margin(y_pred_proba)
    elif strategy == 'least_confident':
        uncertainty = calculate_uncertainty_least_confident(y_pred_proba)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df_with_uncertainty = df.copy()
    df_with_uncertainty['pred_proba'] = y_pred_proba
    df_with_uncertainty['uncertainty'] = uncertainty

    threshold = 0.5
    df_potential_pos = df_with_uncertainty[df_with_uncertainty['pred_proba'] > threshold].copy()
    df_potential_neg = df_with_uncertainty[df_with_uncertainty['pred_proba'] <= threshold].copy()

    n_pos = int(n_with_buffer * positive_ratio)
    n_neg = n_with_buffer - n_pos

    print(f"\n    Stratification:")
    print(f"      Potential positives (proba > {threshold}): {len(df_potential_pos):,} available")
    print(f"      Potential negatives (proba ≤ {threshold}): {len(df_potential_neg):,} available")
    print(f"      Drawing: {n_pos} positives + {n_neg} negatives")

    df_pos_sorted = df_potential_pos.sort_values('uncertainty', ascending=False)
    df_neg_sorted = df_potential_neg.sort_values('uncertainty', ascending=False)

    n_pos_actual = min(n_pos, len(df_pos_sorted))
    n_neg_actual = min(n_neg, len(df_neg_sorted))

    if n_pos_actual < n_pos:
        print(f"       Too few potential positives. Selected {n_pos_actual}/{n_pos}")
        n_neg_actual = min(n_with_buffer - n_pos_actual, len(df_neg_sorted))

    df_selected_pos = df_pos_sorted.head(n_pos_actual)
    df_selected_neg = df_neg_sorted.head(n_neg_actual)

    df_selected = pd.concat([df_selected_pos, df_selected_neg], ignore_index=True)

    df_selected = df_selected.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n    Statistics of the selected samples:")
    print(f"      Total: {len(df_selected)}")
    print(f"      Actual positives: {n_pos_actual} ({n_pos_actual/len(df_selected):.1%})")
    print(f"      Actual negatives: {n_neg_actual} ({n_neg_actual/len(df_selected):.1%})")
    print(f"      Uncertainty: min={df_selected['uncertainty'].min():.4f}, max={df_selected['uncertainty'].max():.4f}, mean={df_selected['uncertainty'].mean():.4f}")
    print(f"      Pred proba: min={df_selected['pred_proba'].min():.4f}, max={df_selected['pred_proba'].max():.4f}, mean={df_selected['pred_proba'].mean():.4f}")

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(df_selected['pred_proba'], bins=bins)

    print(f"\n      pred_proba distribution (selected samples):")
    for i in range(len(bins) - 1):
        pct = hist[i] / len(df_selected) * 100
        print(f"         [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d} samples ({pct:5.1f}%)")

    potential_yes_ratio = (df_selected['pred_proba'] > 0.5).sum() / len(df_selected)
    if potential_yes_ratio < 0.2:
        print(f"\n       WARNING: only {potential_yes_ratio:.1%} potential positives.")
        print(f"         The model may become too conservative in the next iteration")
    elif potential_yes_ratio > 0.8:
        print(f"\n       WARNING: as many as {potential_yes_ratio:.1%} potential positives.")
        print(f"         The model may lose its ability to reject negatives")

    return df_selected, df_with_uncertainty


def analyze_selection(df_selected, df_all, train_score_mean=None):
    print(f"\nAnalysing the selection...")

    if 'score' in df_selected.columns:
        selected_score = df_selected['score'].mean()
        pool_score = df_all['score'].mean()

        print(f"\n   Dictionary score:")
        print(f"      Selected samples: {selected_score:.3f}")
        print(f"      Whole pool:       {pool_score:.3f}")

        if train_score_mean is not None:
            print(f"      Train set:        {train_score_mean:.3f}")

            score_diff_train = selected_score - train_score_mean
            print(f"\n      Difference (selected - train): {score_diff_train:+.3f}")

            if abs(score_diff_train) > 1.0:
                print(f"\n      WARNING: LARGE DISTRIBUTION SHIFT")
                print(f"         The selected score differs substantially from the training set")
                print(f"         The model may be overconfident or underconfident")

                if score_diff_train < -1.0:
                    print(f"\n         → Selection has a LOWER score (diff: {score_diff_train:.3f})")
                    print(f"         → Model was trained on score={train_score_mean:.3f}")
                    print(f"         → It is now selecting samples with score={selected_score:.3f}")
                    print(f"         → RISK: the model may be OVERCONFIDENT")
                    print(f"         → REMEDY: consider adding 'easy positives'")
                    print(f"            (high score + high probability)")
                else:
                    print(f"\n         → Selection has a HIGHER score (diff: {score_diff_train:+.3f})")
                    print(f"         → This is good: searching richer ground for maps")
            elif abs(score_diff_train) > 0.5:
                print(f"       Moderate distribution shift (difference: {score_diff_train:+.3f})")
                print(f"         Monitor performance after this iteration")
            else:
                print(f"      Distribution shift OK (difference: {score_diff_train:+.3f})")

        score_diff_pool = selected_score - pool_score
        print(f"\n      Difference (selected - pool): {score_diff_pool:+.3f}")
        if score_diff_pool > 0:
            print(f"         → Selection scores HIGHER (potentially more maps)")
        elif score_diff_pool < -0.1:
            print(f"         → Selection scores LOWER")
        else:
            print(f"         → Selection is representative of the pool")


def save_iteration(df_selected, iteration_num, strategy, model_package, stats):
    print(f"\nSaving iteration {iteration_num}...")

    iteration_dir = ITERATIONS_DIR / f"iteration_{iteration_num}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    to_annotate = df_selected[['uid', 'url', 'pred_proba']].copy()
    to_annotate_file = iteration_dir / "to_annotate.parquet"
    to_annotate.to_parquet(to_annotate_file, index=False)

    print(f"   Samples for annotation: {to_annotate_file}")
    print(f"      Number of samples: {len(to_annotate)}")

    full_data_file = iteration_dir / "to_annotate_full.parquet"
    df_selected.to_parquet(full_data_file, index=False)

    print(f"   Full data (with uncertainty): {full_data_file}")

    metadata = {
        'iteration': iteration_num,
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df_selected),
        'strategy': strategy,
        'model': {
            'name': model_package['name'],
            'use_smote': model_package['use_smote'],
            'baseline_f1': model_package['results']['f1'],
            'baseline_recall': model_package['results']['recall'],
            'optimal_threshold': model_package['optimal_threshold']
        },
        'stats': stats
    }

    metadata_file = iteration_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   Metadata: {metadata_file}")

    return iteration_dir


def create_annotation_instructions(iteration_dir, n_samples):
    iteration_num = int(iteration_dir.name.split('_')[1])

    instructions = f"""# Annotation instructions - iteration {iteration_num}

## Task
Annotate {n_samples} **VALID** samples as:
- **YES** - a statistical map
- **NO** - not a statistical map
- **INVALID** - dead URL or broken image (does NOT count towards the target)

## Purpose
Active learning selected these samples as the **most uncertain** for the model.
They are borderline cases that should help the model learn.

## How to annotate

Label the records in `to_annotate.parquet` following the labelling criteria in `MODEL_CARD.md` and save
the labels (uid, url, label = YES / NO / INVALID) as `{iteration_dir.name}/annotated.parquet`.
`retrain_model.py --iteration {iteration_num}` reads that file. (The local annotation tool is
not included.)

## Counts

More than {n_samples} samples are drawn as a buffer against invalid URLs.
The expected share of invalid URLs is ~38%, so ~{int(n_samples / 0.62)} samples are drawn.
You only annotate until {n_samples} VALID annotations are reached.
"""

    instructions_file = iteration_dir / "INSTRUCTIONS.md"
    with open(instructions_file, 'w') as f:
        f.write(instructions)

    print(f"   Instructions: {instructions_file}")


def main(n_samples=600, strategy='entropy', iteration_num=1, invalid_rate=0.38,
         positive_ratio=0.4, use_stratified=True):
    print("\n" + "="*80)
    print("ACTIVE LEARNING - SELECTING SAMPLES FOR ANNOTATION")
    print("="*80)

    train_score_mean = None
    if TRAIN_FILE.exists():
        print("\nLoading train statistics...")
        df_train = pd.read_parquet(TRAIN_FILE)
        if 'score' in df_train.columns:
            train_score_mean = df_train['score'].mean()
            print(f"   Train mean score: {train_score_mean:.3f}")

    model_package = load_model()

    df_pool, X_pool = load_pool_data()

    y_pred_proba = predict_probabilities(model_package, X_pool)

    if use_stratified:
        print(f"\nUsing STRATIFIED uncertainty sampling (recommended)")
        df_selected, df_all = select_samples_stratified(
            df_pool, y_pred_proba,
            n_samples=n_samples,
            strategy=strategy,
            invalid_rate=invalid_rate,
            positive_ratio=positive_ratio
        )
    else:
        print(f"\n Using PURE uncertainty sampling (may select edge cases only)")
        df_selected, df_all = select_samples_to_annotate(
            df_pool, y_pred_proba,
            n_samples=n_samples,
            strategy=strategy,
            invalid_rate=invalid_rate
        )

    analyze_selection(df_selected, df_all, train_score_mean=train_score_mean)

    stats = {
        'n_samples_selected': len(df_selected),
        'uncertainty_mean': float(df_selected['uncertainty'].mean()),
        'uncertainty_std': float(df_selected['uncertainty'].std()),
        'pred_proba_mean': float(df_selected['pred_proba'].mean()),
        'pred_proba_std': float(df_selected['pred_proba'].std()),
        'potential_positive_count': int((df_selected['pred_proba'] > 0.5).sum()),
        'potential_positive_ratio': float((df_selected['pred_proba'] > 0.5).mean()),
        'stratified': use_stratified,
        'positive_ratio': positive_ratio if use_stratified else None
    }

    if 'score' in df_selected.columns:
        stats['selected_score_mean'] = float(df_selected['score'].mean())
        stats['pool_score_mean'] = float(df_all['score'].mean())
        if train_score_mean is not None:
            stats['train_score_mean'] = float(train_score_mean)
            stats['distribution_shift'] = float(df_selected['score'].mean() - train_score_mean)

    iteration_dir = save_iteration(df_selected, iteration_num, strategy, model_package, stats)

    create_annotation_instructions(iteration_dir, n_samples)

    print("\n" + "="*80)
    print("ACTIVE LEARNING - DONE")
    print("="*80)
    print(f"\nIteration directory: {iteration_dir}")
    print(f"Samples to annotate: {n_samples} (target VALID)")
    print(f"Samples selected: {len(df_selected)} (including the invalid-URL buffer)")
    print(f"Strategy: {strategy}" + (" (STRATIFIED)" if use_stratified else " (pure)"))
    if use_stratified:
        print(f"   Positive ratio: {positive_ratio:.1%}")

    print(f"\nModel predictions for the selected samples:")
    print(f"   Potential YES (proba > 0.5): {stats['potential_positive_count']} ({stats['potential_positive_ratio']:.1%})")
    print(f"   Mean probability: {stats['pred_proba_mean']:.3f}")

    if 'distribution_shift' in stats:
        print(f"\nDistribution shift:")
        print(f"   Train mean score: {stats['train_score_mean']:.3f}")
        print(f"   Selected mean score: {stats['selected_score_mean']:.3f}")
        print(f"   Shift: {stats['distribution_shift']:+.3f}")
        if abs(stats['distribution_shift']) > 1.0:
            print(f"    NOTE: large distribution shift")

    print(f"\nNext steps:")
    print(f"   1. Annotate the samples using the annotation interface")
    print(f"   2. The interface writes: {iteration_dir}/annotated.parquet")
    print(f"   3. Then run retrain_model.py")
    print(f"   4. Start the next active-learning iteration")

    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Active learning - select samples for annotation')
    parser.add_argument('--n-samples', type=int, default=600,
                       help='Target number of VALID annotations; the interface skips invalid ones automatically (default: 600)')
    parser.add_argument('--strategy', type=str, default='entropy',
                       choices=['entropy', 'margin', 'least_confident'],
                       help='Uncertainty sampling strategy (default: entropy)')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number (default: 1)')
    parser.add_argument('--invalid-rate', type=float, default=0.38,
                       help='Expected %% of invalid URLs, used to size the buffer (default: 0.38 = 38%%)')
    parser.add_argument('--positive-ratio', type=float, default=0.4,
                       help='For stratified sampling: %% of potential positives (default: 0.4 = 40%%)')
    parser.add_argument('--no-stratified', action='store_true',
                       help='Disable stratification and use pure uncertainty (not recommended)')

    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        strategy=args.strategy,
        iteration_num=args.iteration,
        invalid_rate=args.invalid_rate,
        positive_ratio=args.positive_ratio,
        use_stratified=not args.no_stratified
    )
