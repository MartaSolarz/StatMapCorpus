import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sample"
BASELINE_DIR = BASE_DIR / "data" / "baseline"
OUTPUT_DIR = BASE_DIR / "data"

BASELINE_FILE = BASELINE_DIR / "baseline.parquet"
EMBEDDINGS_FILE = DATA_DIR / "sample_200k_with_embeddings.parquet"
METADATA_FILE = DATA_DIR / "sample_200k.parquet"

TRAIN_OUTPUT = OUTPUT_DIR / "train.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test.parquet"
POOL_OUTPUT = OUTPUT_DIR / "pool.parquet"
STATS_OUTPUT = OUTPUT_DIR / "data_preparation_stats.json"


def load_annotations():
    print("Loading baseline annotations...")
    df = pd.read_parquet(BASELINE_FILE)

    print(f"   Number of annotations: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Label distribution:")
    print(df['label'].value_counts().to_string())

    df['label'] = df['label'].map({'YES': 1, 'NO': 0})

    print(f"\n   YES (1): {(df['label'] == 1).sum()}")
    print(f"   NO (0): {(df['label'] == 0).sum()}")
    print(f"   Class balance: {(df['label'] == 1).sum() / len(df):.2%}")

    return df


def load_embeddings():
    print("\nLoading embeddings...")
    df = pd.read_parquet(EMBEDDINGS_FILE)

    print(f"   Samples with embeddings: {len(df)}")
    print(f"   Embedding dimension: {df.iloc[0]['l14_img'].shape}")

    return df


def load_metadata():
    print("\nLoading metadata...")
    df = pd.read_parquet(METADATA_FILE)

    print(f"   Samples with metadata: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    return df


def merge_data(annotations, embeddings, metadata):
    print("\nJoining the data...")

    annotated_data = annotations.merge(embeddings, on='uid', how='inner')
    print(f"   After merging embeddings: {len(annotated_data)} samples")

    metadata_subset = metadata[['uid', 'score']].copy()
    annotated_data = annotated_data.merge(metadata_subset, on='uid', how='left')
    print(f"   After merging metadata: {len(annotated_data)} samples")

    missing_embeddings = len(annotations) - len(annotated_data)
    if missing_embeddings > 0:
        print(f"    No embeddings for {missing_embeddings} samples")

    return annotated_data


def split_train_test(annotated_data, test_size=0.2):
    print(f"\n Train/test split (test_size={test_size}, stratified)...")

    train_data, test_data = train_test_split(
        annotated_data,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=annotated_data['label']
    )

    print(f"   Train: {len(train_data)} samples")
    print(f"      YES (1): {(train_data['label'] == 1).sum()} ({(train_data['label'] == 1).sum()/len(train_data):.2%})")
    print(f"      NO (0): {(train_data['label'] == 0).sum()} ({(train_data['label'] == 0).sum()/len(train_data):.2%})")

    print(f"   Test: {len(test_data)} samples")
    print(f"      YES (1): {(test_data['label'] == 1).sum()} ({(test_data['label'] == 1).sum()/len(test_data):.2%})")
    print(f"      NO (0): {(test_data['label'] == 0).sum()} ({(test_data['label'] == 0).sum()/len(test_data):.2%})")

    print(f"\n    The test set is FIXED. Save the UIDs so it can be verified.")

    return train_data, test_data


def create_pool_data(embeddings, metadata, annotated_uids):
    print("\nBuilding the pool...")

    pool_data = embeddings.merge(metadata, on='uid', how='inner')

    pool_data = pool_data[~pool_data['uid'].isin(annotated_uids)].copy()

    print(f"   Samples in pool: {len(pool_data)}")
    print(f"   Score statistics:")
    if 'score' in pool_data.columns:
        print(f"      Mean: {pool_data['score'].mean():.3f}")
        print(f"      Std: {pool_data['score'].std():.3f}")
        print(f"      Min: {pool_data['score'].min():.3f}, Max: {pool_data['score'].max():.3f}")

    return pool_data


def save_data(train_data, test_data, pool_data):
    print("\nWriting the data...")

    labeled_cols = ['uid', 'label', 'l14_img', 'url', 'score']
    labeled_cols = [c for c in labeled_cols if c in train_data.columns]
    train_data = train_data[labeled_cols].copy()
    test_data = test_data[labeled_cols].copy()

    pool_cols = ['uid', 'l14_img', 'url', 'score']
    pool_cols = [c for c in pool_cols if c in pool_data.columns]
    pool_data = pool_data[pool_cols].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_parquet(TRAIN_OUTPUT, index=False)
    test_data.to_parquet(TEST_OUTPUT, index=False)
    pool_data.to_parquet(POOL_OUTPUT, index=False)

    print(f"   Train data: {TRAIN_OUTPUT}")
    print(f"      Size: {len(train_data)} samples")
    print(f"   Test data: {TEST_OUTPUT}")
    print(f"      Size: {len(test_data)} samples")
    print(f"   Pool data: {POOL_OUTPUT}")
    print(f"      Size: {len(pool_data)} samples")

    test_uids_file = OUTPUT_DIR / "test_uids.txt"
    with open(test_uids_file, 'w') as f:
        for uid in test_data['uid']:
            f.write(f"{uid}\n")
    print(f"   Test UIDs: {test_uids_file}")

    return train_data, test_data, pool_data


def save_statistics(train_data, test_data, pool_data):
    stats = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'train': {
            'size': len(train_data),
            'label_distribution': {
                'YES (1)': int((train_data['label'] == 1).sum()),
                'NO (0)': int((train_data['label'] == 0).sum())
            },
            'class_balance': float((train_data['label'] == 1).sum() / len(train_data)),
            'embedding_dim': int(train_data.iloc[0]['l14_img'].shape[0]),
            'columns': list(train_data.columns)
        },
        'test': {
            'size': len(test_data),
            'label_distribution': {
                'YES (1)': int((test_data['label'] == 1).sum()),
                'NO (0)': int((test_data['label'] == 0).sum())
            },
            'class_balance': float((test_data['label'] == 1).sum() / len(test_data)),
            'embedding_dim': int(test_data.iloc[0]['l14_img'].shape[0]),
            'columns': list(test_data.columns)
        },
        'pool': {
            'size': len(pool_data),
            'embedding_dim': int(pool_data.iloc[0]['l14_img'].shape[0]),
            'columns': list(pool_data.columns)
        }
    }

    for dataset_name, dataset in [('train', train_data), ('test', test_data), ('pool', pool_data)]:
        if 'score' in dataset.columns:
            stats[dataset_name]['score_stats'] = {
                'mean': float(dataset['score'].mean()),
                'std': float(dataset['score'].std()),
                'min': float(dataset['score'].min()),
                'max': float(dataset['score'].max())
            }

    with open(STATS_OUTPUT, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics written: {STATS_OUTPUT}")

    return stats


def display_summary(stats):
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)

    print(f"\nTRAINING DATA:")
    print(f"   Samples: {stats['train']['size']}")
    print(f"   YES (1): {stats['train']['label_distribution']['YES (1)']} ({stats['train']['class_balance']:.2%})")
    print(f"   NO (0): {stats['train']['label_distribution']['NO (0)']} ({1-stats['train']['class_balance']:.2%})")
    print(f"   Embedding dimension: {stats['train']['embedding_dim']}")

    print(f"\nTEST DATA (FIXED SET):")
    print(f"   Samples: {stats['test']['size']}")
    print(f"   YES (1): {stats['test']['label_distribution']['YES (1)']} ({stats['test']['class_balance']:.2%})")
    print(f"   NO (0): {stats['test']['label_distribution']['NO (0)']} ({1-stats['test']['class_balance']:.2%})")
    print(f"   Embedding dimension: {stats['test']['embedding_dim']}")

    print(f"\nPOOL (for active learning):")
    print(f"   Samples: {stats['pool']['size']}")
    print(f"   Embedding dimension: {stats['pool']['embedding_dim']}")

    print(f"\nNOTE:")
    print(f"   - The test set is FIXED (random_seed={stats['random_seed']})")
    print(f"   - Use the same test set in every active-learning iteration")
    print(f"   - The test-set UIDs are written to test_uids.txt")

    print(f"\nReady to train the baseline model and start active learning")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("PREPARING DATA FOR ACTIVE LEARNING")
    print("="*60)

    annotations = load_annotations()
    embeddings = load_embeddings()
    metadata = load_metadata()

    annotated_data = merge_data(annotations, embeddings, metadata)

    train_data, test_data = split_train_test(annotated_data, test_size=0.2)

    all_annotated_uids = pd.concat([train_data['uid'], test_data['uid']])
    pool_data = create_pool_data(embeddings, metadata, all_annotated_uids)

    train_data, test_data, pool_data = save_data(train_data, test_data, pool_data)

    stats = save_statistics(train_data, test_data, pool_data)

    display_summary(stats)


if __name__ == "__main__":
    main()
