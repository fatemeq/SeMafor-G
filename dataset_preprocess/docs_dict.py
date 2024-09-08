import json
from tqdm import tqdm
import pickle

def create_dict():
    with open('docs-dict.pkl', 'wb') as f:
        # reading links
        file = open(f'./links.json', 'r', encoding='utf-8')
        lines = file.readlines()
        file.close()
        del file

        # create hashmap of docs
        docs = {}
        for line in tqdm(lines, total=len(lines), unit='D', unit_scale=True, unit_divisor=1000, desc='Creating map of documents'):
            doc = json.loads(line.strip())
            docs[doc['id']] = doc
        del lines

        pickle.dump(docs, f)

create_dict()
