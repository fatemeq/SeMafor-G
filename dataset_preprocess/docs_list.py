import json
from tqdm import tqdm
import pickle

def create_dict():
    with open('docs-list.pkl', 'wb') as f:
        # reading links
        file = open(f'./links.json', 'r', encoding='utf-8')
        lines = file.readlines()
        file.close()
        del file

        # create map of docs
        docs = []
        for idx, line in tqdm(enumerate(lines, 0), total=len(lines), unit='D', unit_scale=True, unit_divisor=1000, desc='Creating documents list'):
            doc = json.loads(line.strip())
            doc['idx'] = idx
            docs.append(json.dumps(doc))

        pickle.dump(docs, f)

create_dict()
