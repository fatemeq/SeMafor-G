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

    docs_file = open('docs-list.pkl', 'rb')
    docs = pickle.load(docs_file)
    docs_file.close()

    # create lsh
    for raw_doc in tqdm(docs, total=len(docs), unit='D', unit_scale=True, unit_divisor=1000, desc='Creating LSH'):
        doc = json.loads(raw_doc)
        m = MinHash(num_perm=128)
        m.update_batch(d.encode('utf8') for d in doc['links'])
        lsh.insert(doc['idx'], m)

    # getting similar docs
    with open('./similar_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['doc1_id', 'doc2_id', 'score'])

        for raw_doc in tqdm(docs, total=len(docs), unit='D', unit_scale=True, unit_divisor=1000, desc='Calculating Doc Similarity'):
            doc = json.loads(raw_doc)
            m = MinHash(num_perm=128)
            m.update_batch(d.encode('utf8') for d in doc['links'])
            res = lsh.query(m)
            for idx in res:
                if int(doc['idx']) >= int(idx):
                    continue
                doc2 = json.loads(docs[idx])
                j_score = jacard_score(doc2['links'], doc['links'])
                if j_score >= 0.5:
                    csv_writer.writerow([doc['id'], doc2['id'], j_score])

read_and_calculate_scores()
