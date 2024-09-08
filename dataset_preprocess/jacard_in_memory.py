import time
import os
import json
import sys
import csv
from tqdm import tqdm

def jacard_score(links1, links2):
    union = set(links1 + links2)
    intersection = set(links1) & set(links2)
    
    return len(intersection) / len(union)


def read_and_calculate_scores():
    with open('./similar_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['doc1_id', 'doc2_id', 'score'])
        
        print('loading file')
        links_file = open(f'./links.json', 'r', encoding='utf-8')
        links_lines = links_file.readlines()
        links_file.close()
        print('file loaded')

        for line_idx, line in tqdm(enumerate(links_lines, 0), total=len(links_lines), unit='Doc', unit_scale=True, unit_divisor=1000, desc='Source Processing'):
            print('loading first doc')
            doc1 = json.loads(line.strip())
            for line in tqdm(links_lines[line_idx:], unit='Doc', total=len(links_lines[line_idx:]), unit_scale=True, unit_divisor=1000, desc='Target Processing', leave=False):
                doc2 = json.loads(line.strip())

                if doc1['id'] == doc2['id']:
                    continue
                score = jacard_score(doc1['links'], doc2['links'])
                if score > 0.5:
                    csv_writer.writerow([doc1['id'], doc2['id'], score])

read_and_calculate_scores()

