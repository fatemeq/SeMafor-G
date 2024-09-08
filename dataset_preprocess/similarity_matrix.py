import os
import json
import sys
import csv
from tqdm import tqdm

def jacard_score(links1, links2):
    union = set(links1 + links2)
    intersection = set(links1) & set(links2)
    
    return len(intersection) / len(union)

MAX_SEGMENT_ID = 31


def get_total_segments_length(start_segment = 1):
    size = 0
    for segment1_id in range(start_segment, MAX_SEGMENT_ID):
        size += os.path.getsize(f'./dataset/links/{segment1_id}.json')
    return size

def read_and_calculate_scores():
    with open('./dataset/similar_docs.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['doc1_id', 'doc2_id', 'score'])
        
        source_doc_progress_bar = tqdm(total=get_total_segments_length(1), unit='B', unit_scale=True, unit_divisor=1024, desc='Source Processing') 
        
        for segment1_id in range(1, MAX_SEGMENT_ID):
            with open(f'./dataset/links/{segment1_id}.json', 'r', encoding='utf-8') as file1:
                for line_number, line in enumerate(file1, 1):
                    doc1 = json.loads(line.strip())

                    source_doc_progress_bar.update(len(line))
                    source_doc_progress_bar.set_description(f'Source Processing Segment: {segment1_id} with Doc: {line_number}')
                    target_doc_progress_bar = tqdm(total=get_total_segments_length(segment1_id), unit='B', unit_scale=True, unit_divisor=1024, desc='Target Processing') 
                    for segment_id in range(segment1_id, MAX_SEGMENT_ID):
                        with open(f'./dataset/links/{segment_id}.json', 'r', encoding='utf-8') as file2:
                            for line_number, line in enumerate(file2, 1):

                                doc2 = json.loads(line)
                                if doc1['id'] == doc2['id']:
                                    continue

                                score = jacard_score(doc1['links'], doc2['links'])
                                if score > 0.5:
                                    csv_writer.writerow([doc1['id'], doc2['id'], score])

                                target_doc_progress_bar.update(len(line))

read_and_calculate_scores()