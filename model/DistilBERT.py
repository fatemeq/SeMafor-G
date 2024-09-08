import os
import json
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, DistilBertTokenizer
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def get_doc_texts():
    docs = {}

    with open('../dataset/docs.json', 'r') as docs_file:
        progress_bar = tqdm(total=os.path.getsize('../dataset/docs.json'), unit='B', unit_scale=True, unit_divisor=1024, desc='Loading docs') 

        for line in docs_file:
            doc = json.loads(line)
            docs[doc['id']] = doc['text']
            
            progress_bar.update(len(line))

    return docs

def get_dataset():
    doc_texts = get_doc_texts()

    docs_df = pd.read_csv("../dataset/docs.csv").astype('string')

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
    print("Loading DistilBERT model")
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    print("Loading dataset")
    train, test = get_dataset()

    print(f'len of train {len(train)}')
    print(f'len of test {len(test)}')

    print("Evaluating DistilBERT model")
    evaluate_distilbert(train, test, tokenizer, distilbert_model)

def evaluate_distilbert(train, test, tokenizer, distilbert_model):
    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Document1', 'Document2', 'ActualScore', 'PredictedScore'])

    # Iterate through each pair in the test set and calculate similarity scores using DistilBERT
    for idx, pair in enumerate(test):
        document1 = pair["doc1"]["text"]
        document2 = pair["doc2"]["text"]
        actual_score = pair["score"]

        # Tokenize and get embeddings for documents using DistilBERT
        inputs1 = tokenizer(document1, return_tensors='pt', max_length=512, truncation=True)
        inputs2 = tokenizer(document2, return_tensors='pt', max_length=512, truncation=True)

        with torch.no_grad():
            output1 = distilbert_model(**inputs1)
            output2 = distilbert_model(**inputs2)

        embeddings1 = output1.last_hidden_state.mean(dim=1)
        embeddings2 = output2.last_hidden_state.mean(dim=1)

        # Calculate the cosine similarity between the embeddings
        cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()

        # Store the results in the results DataFrame
        results_df = results_df.append({'Document1': pair["doc1"]["id"], 'Document2': pair["doc2"]["id"], 'ActualScore': actual_score, 'PredictedScore': cosine_similarity}, ignore_index=True)

        print(f'Similarity Score for pair {idx + 1} ({pair["doc1"]["id"]}, {pair["doc2"]["id"]}): Actual: {actual_score}, Predicted: {cosine_similarity}')

    # Save the results DataFrame to a new CSV file
    results_csv_path = "../dataset/results_distilbert.csv"
    results_df.to_csv(results_csv_path, index=False)

    # Assuming df is your DataFrame containing the results
    actual_scores = results_df['ActualScore'].values
    predicted_scores = results_df['PredictedScore'].values

    # Calculate metrics
    mse = mean_squared_error(actual_scores, predicted_scores)
    mae = mean_absolute_error(actual_scores, predicted_scores)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_scores, predicted_scores)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

if __name__ == "__main__":
    evaluate()
