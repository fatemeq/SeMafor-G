import os
import json
from tqdm import tqdm
import pandas as pd
from model.model import SemanticSimilarityModel, Summerizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def get_doc_texts():
    docs = {}

    with open('./dataset/docs.json', 'r') as docs_file:
        progress_bar = tqdm(total=os.path.getsize('./dataset/docs.json'), unit='B', unit_scale=True, unit_divisor=1024, desc='Loading docs') 

        for line in docs_file:
            doc = json.loads(line)
            docs[doc['id']] = doc['text']
            
            progress_bar.update(len(line))

    return docs

def get_dataset():
    doc_texts = get_doc_texts()

    docs_df = pd.read_csv("./dataset/docs.csv").astype('string')

    batch = []
    for index, row in docs_df.iterrows():
        if not row.doc1_id in doc_texts or not row.doc2_id in doc_texts:
            continue

        batch.append({
            "doc1": {
                "id": row.doc1_id,
                "text": doc_texts[row.doc1_id]
            },
            "doc2": {
                "id": row.doc2_id,
                "text": doc_texts[row.doc2_id]
            },
            "score": float(row.score)
        })

    train, test = train_test_split(batch, test_size=0.2, random_state=42)
    return train, test

def evaluate():
    print("Loading model")
    model = SemanticSimilarityModel()

    print("Loading dataset")
    train, test = get_dataset()

    print(f'len of train {len(train)}')
    print(f'len of test {len(test)}')

    model.load("with_classifier")

    print("Evaluating")
    model.evaluate_batch(test)

def get_dataset_info():
    # Generate a list of numbers
    train, test = get_dataset()

    # Create a scatter plot
    plt.hist([int(t['score'] > 0.5 )for t in test])

    # Label the axes
    plt.xlabel('Data Point')
    plt.ylabel('Value')

    # Set the title
    plt.title('Scatter Plot of Generated Data')

    # Show the plot
    plt.show()
    
# get_dataset_info()    
evaluate()
