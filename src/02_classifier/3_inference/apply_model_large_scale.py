#!/usr/bin/env python3

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import gc
import json
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')


def load_model(model_path):
    print(f"Loading model: {model_path}")
    model_package = joblib.load(model_path)

    print(f"   Model: {model_package['name']}")
    print(f"   Iteration: {model_package.get('iteration', 'baseline')}")
    print(f"   Optimal threshold: {model_package['optimal_threshold']:.4f}")

    return model_package


def process_single_file(args_tuple):
    file_path, model_path, threshold, batch_size, output_dir, file_idx = args_tuple

    try:
        output_file = output_dir / f"predictions_{file_path.stem}.parquet"

        if output_file.exists():
            return {
                'file': file_path.name,
                'status': 'skipped',
                'reason': 'already_processed'
            }

        parquet_file = pq.ParquetFile(file_path)
        total_samples = parquet_file.metadata.num_rows

        print(f"   [Worker {file_idx}] Processing {file_path.name} ({total_samples:,} samples)...")
        print(f"   [Worker {file_idx}] Using STREAMING mode (memory efficient)")

        model_package = joblib.load(model_path)
        model = model_package['model']
        scaler = model_package['scaler']

        print(f"   [Worker {file_idx}] Model loaded, streaming batches...")

        columns_to_read = ['uid', 'url', 'l14_img', 'score']

        predictions = []
        batch_idx = 0
        n_batches = (total_samples + batch_size - 1) // batch_size

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns_to_read):
            if batch_idx % 5 == 0:
                progress_pct = (batch_idx / n_batches) * 100
                print(f"   [Worker {file_idx}] Batch {batch_idx+1}/{n_batches} ({progress_pct:.0f}%)")

            df_batch = batch.to_pandas()

            X_batch = np.stack(df_batch['l14_img'].values)

            if scaler:
                X_batch = scaler.transform(X_batch)

            y_pred_proba = model.predict_proba(X_batch)[:, 1]

            mask = y_pred_proba >= threshold

            if mask.sum() > 0:
                batch_results = pd.DataFrame({
                    'uid': df_batch['uid'].values[mask],
                    'url': df_batch['url'].values[mask],
                    'pred_proba': y_pred_proba[mask]
                })

                if 'score' in df_batch.columns:
                    batch_results['score'] = df_batch['score'].values[mask]

                predictions.append(batch_results)

            del X_batch, y_pred_proba, mask, df_batch, batch
            gc.collect()

            batch_idx += 1

        if predictions:
            df_predictions = pd.concat(predictions, ignore_index=True)
            df_predictions.to_parquet(output_file, index=False)
            n_positive = len(df_predictions)
        else:
            n_positive = 0

        positive_rate = n_positive / total_samples if total_samples > 0 else 0
        print(f"   [Worker {file_idx}] Done: {n_positive:,}/{total_samples:,} ({positive_rate:.2%}) positive")

        return {
            'file': file_path.name,
            'status': 'success',
            'total': total_samples,
            'positive': n_positive,
            'positive_rate': positive_rate,
            'output': str(output_file)
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"   [Worker {file_idx}] ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            'file': file_path.name,
            'status': 'error',
            'reason': error_msg,
            'total': 0
        }


def find_parquet_files(input_dir):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

    parquet_files = sorted(input_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in: {input_dir}")

    return parquet_files


def merge_prediction_files(temp_dir, output_file, keep_temp=False):
    print(f"\nMerging results...")

    temp_path = Path(temp_dir)
    prediction_files = sorted(temp_path.glob("predictions_*.parquet"))

    if not prediction_files:
        print("    Nothing to merge; every file may have had zero positives")
        return None

    print(f"   Found {len(prediction_files)} files to merge")

    dfs = []
    for pred_file in tqdm(prediction_files, desc="   Merging"):
        try:
            df = pd.read_parquet(pred_file)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"    Failed to read {pred_file.name}: {e}")

    if not dfs:
        print("    Every file was empty")
        return None

    df_final = pd.concat(dfs, ignore_index=True)

    initial_len = len(df_final)
    df_final = df_final.drop_duplicates(subset='uid', keep='first')
    duplicates_removed = initial_len - len(df_final)

    if duplicates_removed > 0:
        print(f"    Dropped {duplicates_removed} duplicates")

    df_final = df_final.sort_values('pred_proba', ascending=False).reset_index(drop=True)

    df_final.to_parquet(output_file, index=False)

    print(f"   Written: {output_file}")
    print(f"   Total predictions: {len(df_final):,}")

    if not keep_temp:
        print(f"\n Removing temporary files...")
        for pred_file in prediction_files:
            pred_file.unlink()
        print(f"   Removed {len(prediction_files)} files")

    return df_final


def save_statistics(stats_list, output_file, elapsed_time):
    stats_file = output_file.parent / f"{output_file.stem}_stats.json"

    total_files = len(stats_list)
    successful = sum(1 for s in stats_list if s['status'] == 'success')
    errors = sum(1 for s in stats_list if s['status'] == 'error')
    skipped = sum(1 for s in stats_list if s['status'] == 'skipped')

    total_samples = sum(s.get('total', 0) for s in stats_list if 'total' in s)
    total_positive = sum(s.get('positive', 0) for s in stats_list if 'positive' in s)

    overall_positive_rate = total_positive / total_samples if total_samples > 0 else 0

    statistics = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_time_seconds': elapsed_time,
        'files': {
            'total': total_files,
            'successful': successful,
            'errors': errors,
            'skipped': skipped
        },
        'samples': {
            'total': total_samples,
            'positive': total_positive,
            'positive_rate': overall_positive_rate
        },
        'throughput': {
            'samples_per_second': total_samples / elapsed_time if elapsed_time > 0 else 0,
            'files_per_minute': (total_files / elapsed_time) * 60 if elapsed_time > 0 else 0
        },
        'per_file_stats': stats_list
    }

    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)

    print(f"\nStatistics written: {stats_file}")

    return statistics


def print_summary(stats):
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)

    print(f"\nFiles:")
    print(f"   Total:      {stats['files']['total']}")
    print(f"   Successful: {stats['files']['successful']} ")
    print(f"   Errors:     {stats['files']['errors']} {'' if stats['files']['errors'] > 0 else ''}")
    print(f"   Skipped:    {stats['files']['skipped']}")

    print(f"\nSamples:")
    print(f"   Total processed: {stats['samples']['total']:,}")
    print(f"   Positive (>= threshold): {stats['samples']['positive']:,}")
    print(f"   Positive rate: {stats['samples']['positive_rate']:.2%}")

    print(f"\n Performance:")
    print(f"   Elapsed time: {stats['elapsed_time_seconds']:.1f}s ({stats['elapsed_time_seconds']/60:.1f} min)")
    print(f"   Throughput: {stats['throughput']['samples_per_second']:.0f} samples/sec")
    print(f"   Files/min: {stats['throughput']['files_per_minute']:.1f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Apply model large scale (50M+ samples)')

    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory of Parquet files to process')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Path of the output file holding the final predictions')
    parser.add_argument('--model', type=str,
                       default=str(Path(__file__).resolve().parents[1] / 'models' / 'model_final.pkl'),
                       help='Path to the model (default: the published models/model_final.pkl)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='pred_proba threshold; only rows >= threshold are written (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Prediction batch size (default: 10000)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers, i.e. files processed in parallel (default: 4)')
    parser.add_argument('--temp-dir', type=str, default=None,
                       help='Temporary directory (default: next to output-file)')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep the temporary files')
    parser.add_argument('--resume', action='store_true',
                       help='Resume, skipping files that have already been processed')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("LARGE SCALE MODEL APPLICATION")
    print("="*80)

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    model_path = Path(args.model)

    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = output_file.parent / f"{output_file.stem}_temp"

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"   Input dir:    {input_dir}")
    print(f"   Output file:  {output_file}")
    print(f"   Model:        {model_path}")
    print(f"   Threshold:    {args.threshold}")
    print(f"   Batch size:   {args.batch_size:,}")
    print(f"   Workers:      {args.workers}")
    print(f"   Temp dir:     {temp_dir}")
    print(f"   Resume:       {args.resume}")

    print(f"\nLooking for Parquet files...")
    parquet_files = find_parquet_files(input_dir)
    print(f"   Found: {len(parquet_files)} files")

    total_size_mb = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
    print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")

    model_package = load_model(model_path)

    tasks = [
        (file_path, model_path, args.threshold, args.batch_size, temp_dir, idx)
        for idx, file_path in enumerate(parquet_files)
    ]

    print(f"\n Processing ({args.workers} workers)...")
    print(f"   ℹ Each worker loads the model independently")
    start_time = datetime.now()

    stats_list = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}

        with tqdm(total=len(parquet_files), desc="Files processed") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats_list.append(result)

                status_emoji = {
                    'success': '',
                    'error': '',
                    'skipped': ''
                }.get(result['status'], '')

                pbar.set_postfix_str(f"{result['file']} {status_emoji}")
                pbar.update(1)

                if result['status'] == 'error' and 'reason' in result:
                    print(f"\n{result['file']}: {result['reason']}")

    elapsed_time = (datetime.now() - start_time).total_seconds()

    print(f"\nMerging results from {len(parquet_files)} files...")
    df_final = merge_prediction_files(temp_dir, output_file, keep_temp=args.keep_temp)

    statistics = save_statistics(stats_list, output_file, elapsed_time)

    print_summary(statistics)

    if df_final is not None:
        print(f"\nDONE")
        print(f"   Results: {output_file}")
        print(f"   Predictions: {len(df_final):,}")
        print(f"   Positive rate: {statistics['samples']['positive_rate']:.2%}")
        print(f"   Elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    else:
        print(f"\n No positive predictions; every score was < {args.threshold}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
