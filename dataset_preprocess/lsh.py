import pickle
import gc
import csv
import os
import json
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

def jacard_score(links1, links2):
    union = set(links1 + links2)
    intersection = set(links1) & set(links2)
    return len(intersection) / len(union)

def read_and_calculate_scores():
    lsh = MinHashLSH(threshold=0.6, num_perm=128)

    docs_file = open('docs-dict.pkl', 'rb')
    docs = pickle.load(docs_file)
    docs_file.close()

    # create lsh
    for doc in tqdm(docs.values(), total=len(docs), unit='D', unit_scale=True, unit_divisor=1000, desc='Creating LSH'):
        m = MinHash(num_perm=128)
        m.update_batch(d.encode('utf8') for d in doc['links'])
        lsh.insert(doc['id'], m)

    # getting similar docs
    with open('./similar_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['doc1_id', 'doc2_id', 'score'])

        for doc in tqdm(docs.values(), total=len(docs), unit='D', unit_scale=True, unit_divisor=1000, desc='Calculating Doc Similarity'):
            m = MinHash(num_perm=128)
            m.update_batch(d.encode('utf8') for d in doc['links'])
            res = lsh.query(m)
            for id in res:
                if int(doc['id']) >= int(id):
                    continue
                j_score = jacard_score(docs[id], doc)
                if j_score >= 0.5:
                    csv_writer.writerow([doc['id'], id, j_score])

read_and_calculate_scores()
