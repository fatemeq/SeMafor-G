import csv
from tqdm import tqdm
import os
import json

def unique_document_ids(csv_file_path):
    document_ids = set()

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row if exists

        for row in csv_reader:
            # Assuming the IDs are in the first and second columns
            id1, id2, _ = row
            document_ids.add(id1)
            document_ids.add(id2)


    return document_ids

def extract_docs(input_file_path):
    unique_ids = unique_document_ids('./dataset/small_similar_docs.csv')

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open('./dataset/similar_docs.json', 'w', encoding='utf-8') as output_file:
            pb = tqdm(total=os.path.getsize(input_file_path), unit='B', unit_scale=True, unit_divisor=1024, desc='Extracting docs')
            
            current_doc = None
            for line in input_file:
                pb.update(len(line))

                if current_doc != None and current_doc["text_tag_started"] == False and line.lstrip().startswith("<text"):
                    current_doc["text"] = line.strip() # TODO: replace tag
                    current_doc["text_tag_started"] = True
                    continue

                if current_doc != None and current_doc["text_tag_started"] == True:
                    if line.rstrip().endswith("</text>"):
                        current_doc["text"] += line.strip().replace('</text>', '')
                        
                        output_file.write(json.dumps({ 'id': current_doc["id"], 'text': current_doc["text"] }))
                        output_file.write('\n')

                        current_doc = None
                    else:
                        current_doc["text"] += line
                    continue

                if current_doc == None and line.lstrip().startswith("<id>"):
                    for doc_id in unique_ids:
                        if f"<id>{str(doc_id)}</id>" in line:
                            current_doc = {
                                "id": str(doc_id),
                                "text": "",
                                "text_tag_started": False
                            }
                            unique_ids.discard(doc_id)
                            break


def extract_non_similar_docs(input_file_path):
    unique_ids = unique_document_ids('./dataset/non_similar_docs.csv')

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open('./dataset/non_similar_docs.json', 'w', encoding='utf-8') as output_file:
            pb = tqdm(total=os.path.getsize(input_file_path), unit='B', unit_scale=True, unit_divisor=1024, desc='Extracting docs')
            
            current_doc = None
            for line in input_file:
                if len(unique_ids) == 0:
                    break

                pb.update(len(line))

                if current_doc != None and current_doc["text_tag_started"] == False and line.lstrip().startswith("<text"):
                    current_doc["text"] = line.strip() # TODO: replace tag
                    current_doc["text_tag_started"] = True
                    continue

                if current_doc != None and current_doc["text_tag_started"] == True:
                    if line.rstrip().endswith("</text>"):
                        current_doc["text"] += line.strip().replace('</text>', '')
                        
                        output_file.write(json.dumps({ 'id': current_doc["id"], 'text': current_doc["text"] }))
                        output_file.write('\n')

                        current_doc = None
                    else:
                        current_doc["text"] += line
                    continue

                if current_doc == None and line.lstrip().startswith("<id>"):
                    for doc_id in unique_ids:
                        if f"<id>{str(doc_id)}</id>" in line:
                            current_doc = {
                                "id": str(doc_id),
                                "text": "",
                                "text_tag_started": False
                            }
                            unique_ids.discard(doc_id)
                            break
                



# extract_docs('./dataset/enwiki-latest-pages-articles.xml')
extract_non_similar_docs('./dataset/enwiki-latest-pages-articles.xml')
