import os
import json
import shutil
import argparse
import ahocorasick
import threading
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from collections import defaultdict

BASE_DIR = Path(__file__).parent
KEYWORDS_DIR = BASE_DIR / "keywords"
SHARDS_FILE = BASE_DIR / "mappool_shards.json"
DOWNLOAD_FOLDER = "./files_all"
OUTPUT_FOLDER = "./processed"
BATCH_SIZE = 1000
MAX_WORKERS = 10
REPO_ID = "sraimund/MapPool"
TOKEN = os.environ.get("HF_TOKEN")

file_locks = {}
batch_counters = {}

def compute_score_full_words(text, automaton):
    if not isinstance(text, str):
        text = str(text)
    text_lower = text.lower()
    score = 0
    used_phrases = set()
    pos_keys, neg_keys = [], []

    for end_index, (ptype, phrase, value) in automaton.iter(text_lower):
        start_index = end_index - len(phrase) + 1
        if (start_index == 0 or not text_lower[start_index-1].isalnum()) and \
           (end_index == len(text_lower)-1 or not text_lower[end_index+1].isalnum()):
            if phrase not in used_phrases:
                score += value
                used_phrases.add(phrase)
                if ptype == "pos":
                    pos_keys.append(phrase)
                else:
                    neg_keys.append(phrase)
    return score, pos_keys, neg_keys


def get_file_lock(file_path):
    if file_path not in file_locks:
        file_locks[file_path] = threading.Lock()
    return file_locks[file_path]


def save_batch(batch_records, output_file):
    records_without_emb = []
    embeddings_list = []

    for record in batch_records:
        rec_copy = {k: v for k, v in record.items() if k != 'l14_img'}
        records_without_emb.append(rec_copy)
        embeddings_list.append(np.array(record['l14_img'], dtype=np.float32))

    df_batch = pd.DataFrame(records_without_emb)
    df_batch["uid"] = df_batch["uid"].astype(str)

    df_batch['l14_img'] = embeddings_list

    emb_dim = len(embeddings_list[0]) if embeddings_list else 768

    schema = pa.schema([
        ('uid', pa.string()),
        ('url', pa.string()),
        ('score', pa.float64()),
        ('pos_keys', pa.string()),
        ('neg_keys', pa.string()),
        ('l14_img', pa.list_(pa.float32(), emb_dim))
    ])

    embeddings_array = pa.array(embeddings_list, type=pa.list_(pa.float32(), emb_dim))

    table = pa.table({
        'uid': df_batch['uid'],
        'url': df_batch['url'],
        'score': df_batch['score'],
        'pos_keys': df_batch['pos_keys'],
        'neg_keys': df_batch['neg_keys'],
        'l14_img': embeddings_array
    }, schema=schema)

    output_dir = output_file.replace('.parquet', '_parts')
    os.makedirs(output_dir, exist_ok=True)

    lock = get_file_lock(output_file)
    with lock:
        if output_file not in batch_counters:
            batch_counters[output_file] = 0

        batch_num = batch_counters[output_file]
        batch_counters[output_file] += 1

        batch_file = os.path.join(output_dir, f"part_{batch_num:05d}.parquet")
        pq.write_table(table, batch_file, compression='gzip')


def consolidate_file_parts(output_file):
    output_dir = output_file.replace('.parquet', '_parts')

    if not os.path.exists(output_dir):
        return

    part_files = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith('part_') and f.endswith('.parquet')
    ])

    if not part_files:
        return

    tables = [pq.read_table(f) for f in part_files]
    combined_table = pa.concat_tables(tables)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pq.write_table(combined_table, output_file, compression='gzip')

    shutil.rmtree(output_dir)


def download_file(part, file_name):
    file_path = os.path.join(DOWNLOAD_FOLDER, part, f"{file_name}.parquet")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        return file_path
    try:
        return hf_hub_download(
            repo_type="dataset",
            repo_id=REPO_ID,
            filename=f"{part}/{file_name}.parquet",
            local_dir=DOWNLOAD_FOLDER,
            token=TOKEN
        )
    except Exception as e:
        raise Exception(f"Failed to download {part}/{file_name}: {e}")

def process_file(file_info, automaton):
    part, file_name = file_info
    try:
        file_path = download_file(part, file_name)

        output_file = os.path.join(OUTPUT_FOLDER, part, f"{file_name}_processed.parquet")

        df = pq.read_table(file_path, columns=["uid", "text", "url", "l14_img"]).to_pandas()

        batch_records = []

        for idx, row in df.iterrows():
            score, pos, neg = compute_score_full_words(row["text"], automaton)

            embedding = row["l14_img"]
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)

            record = {
                "uid": row["uid"],
                "url": row["url"],
                "score": score,
                "pos_keys": '|'.join(pos) if pos else '',
                "neg_keys": '|'.join(neg) if neg else '',
                "l14_img": embedding  # numpy array
            }
            batch_records.append(record)

            if len(batch_records) >= BATCH_SIZE:
                save_batch(batch_records, output_file)
                batch_records = []

        if batch_records:
            save_batch(batch_records, output_file)

        consolidate_file_parts(output_file)

        return part, file_name, "success", output_file
    except Exception as e:
        return part, file_name, "error", str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply the keyword dictionaries to the MapPool shards')
    parser.add_argument('--parts', type=str, default=None,
                        help='Parts to process, e.g. "0,1,2", or "all" (default: all)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Number of workers (default: {MAX_WORKERS})')
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    with open(KEYWORDS_DIR / "positive_keywords.json", "r", encoding="utf-8") as f:
        positive_json = json.load(f)
    with open(KEYWORDS_DIR / "negative_keywords.json", "r", encoding="utf-8") as f:
        negative_json = json.load(f)

    print(f"Loaded {len(positive_json)} positive and {len(negative_json)} negative key phrases.")
    print("Building the Aho-Corasick automaton...")

    A = ahocorasick.Automaton()
    for phrase, value in positive_json.items():
        A.add_word(phrase.lower(), ("pos", phrase, value))
    for phrase, value in negative_json.items():
        A.add_word(phrase.lower(), ("neg", phrase, value))
    A.make_automaton()

    with open(SHARDS_FILE, encoding="utf-8") as f:
        file_list_json = json.load(f)

    if args.parts and args.parts.lower() != 'all':
        selected_parts = [f"part_{p.strip()}" for p in args.parts.split(',')]
        file_list_json = {k: v for k, v in file_list_json.items() if k in selected_parts}
        print(f"Processing the selected parts: {', '.join(selected_parts)}")
    else:
        print(f"Processing every part: {', '.join(file_list_json.keys())}")

    all_files = [(part, file_name) for part, files in file_list_json.items() for file_name in files]
    total_files = len(all_files)
    print(f"Found {total_files} shards to process.")

    num_workers = args.workers
    print(f"Starting with {num_workers} workers (ThreadPoolExecutor)...")
    success_count = 0
    error_count = 0

    processed_parts = defaultdict(set)
    total_parts = defaultdict(int)

    for part, file_name in all_files:
        total_parts[part] += 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f, A): f for f in all_files}
        for future in tqdm(as_completed(futures), total=total_files, desc="Processing shards"):
            part, file_name, status, result = future.result()
            if status == "success":
                success_count += 1
                processed_parts[part].add(file_name)

                if len(processed_parts[part]) == total_parts[part]:
                    part_folder = os.path.join(DOWNLOAD_FOLDER, part)
                    if os.path.exists(part_folder):
                        try:
                            shutil.rmtree(part_folder)
                            print(f"\n  Removed the folder of downloaded shards: {part}/")
                        except Exception as e:
                            print(f"\n️  Could not remove the folder {part}/: {e}")
            else:
                error_count += 1
                print(f"\n Error in {part}/{file_name}: {result}")

    print(f"\n Finished.")
    print(f"   Succeeded: {success_count}/{total_files}")
    print(f"   Failed:    {error_count}/{total_files}")
