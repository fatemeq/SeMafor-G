import csv
import os
import json
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

def jacard_score(links1, links2):
    union = set(links1 + links2)
    intersection = set(links1) & set(links2)
    return len(intersection) / len(union)

MAX_SEGMENT_ID = 31

def get_total_segments_length(start_segment = 1):
    size = 0
    for segment_id in range(start_segment, MAX_SEGMENT_ID):
        size += os.path.getsize(f'./dataset/links/{segment_id}.json')
    return size

def read_and_calculate_scores():
    lsh = MinHashLSH(threshold=0.6, num_perm=128)

    source_doc_progress_bar = tqdm(total=get_total_segments_length(1), unit='B', unit_scale=True, unit_divisor=1024, desc='Source Processing')

    # calculating lsh
    for segment_id in range(1, MAX_SEGMENT_ID):
        with open(f'./dataset/links/{segment_id}.json', 'r', encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line.strip())
                m = MinHash(num_perm=128)
                m.update_batch(d.encode('utf8') for d in doc['links'])
                lsh.insert(doc['id'], m)

                source_doc_progress_bar.update(len(line))

    source_doc_progress_bar.clear()

    # getting similar docs
    with open('./dataset/similar_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['doc1_id', 'doc2_id', 'score'])

        for segment_id in range(1, MAX_SEGMENT_ID):
            with open(f'./dataset/links/{segment_id}.json', 'r', encoding='utf-8') as file:
                for line in file:
                    doc = json.loads(line.strip())
                    m = MinHash(num_perm=128)
                    m.update_batch(d.encode('utf8') for d in doc['links'])
                    res = lsh.query(m)
                    for i in res:
                        if i == doc['id']:
                            continue
                        csv_writer.writerow([doc['id'], i, 1])
                    source_doc_progress_bar.update(len(line))

read_and_calculate_scores()

