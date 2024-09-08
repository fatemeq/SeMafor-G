import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import networkx as nx
import os

from transformers import AutoTokenizer, AutoModel
from transformers import RobertaModel, RobertaTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, InputExample, losses

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv

class Summerizer:
    def __init__(self):
        model_name = 't5-base' # ['t5-base', 'facebook/bart-large-cnn', 'google/pegasus-xsum']
        self.tokenizer = T5Tokenizer.from_pretrained(model_name,  model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        # self.tokenizer = BartTokenizer.from_pretrained(model_name)
        # self.model = BartForConditionalGeneration.from_pretrained(model_name)
        # self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        # self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

    def summerize(self, text):
        # Tokenize and generate summary
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(input_ids, min_length=60, max_length=180, length_penalty=4.0)

        # Decode and return the generated summary
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class LSA:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        num_topics = 2
        self.lsa = TruncatedSVD(n_components=num_topics, random_state=42)
        self.problematic_docs = []  # Store problematic documents

    def encode(self, doc):
        if not doc:
            return torch.tensor([])
        try:
            tfidf_matrix = self.vectorizer.fit_transform([doc])

            if not self.vectorizer.vocabulary_:
                # Handle empty vocabulary
                return torch.tensor([])

            # Check if all features in the vocabulary are stop words
            if all(word in self.vectorizer.get_feature_names_out() for word in self.vectorizer.get_stop_words()):
                # Handle case where documents only contain stop words
                return torch.tensor([])

            if tfidf_matrix.shape[1] > 1:
                tfidf_matrix = self.lsa.fit_transform(tfidf_matrix)

            if type(tfidf_matrix[0]) is np.ndarray:
                return torch.from_numpy(tfidf_matrix[0]).reshape(-1)

            return torch.tensor(tfidf_matrix[0].toarray()).reshape(-1)

        except Exception as e:
            # Print the exception and store the problematic documents
            print(f"Error processing documents: {e}")
            print("Doc:", doc)
            self.problematic_docs.append(doc)
            return torch.tensor([])

    def score(self, doc1, doc2):
        if not doc1 or not doc2:
            # Handle empty documents
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([doc1, doc2])

            if not self.vectorizer.vocabulary_:
                # Handle empty vocabulary
                return 0.0

            # Check if all features in the vocabulary are stop words
            if all(word in self.vectorizer.get_feature_names_out() for word in self.vectorizer.get_stop_words()):
                # Handle case where documents only contain stop words
                return 0.0

            if tfidf_matrix.shape[1] > 1:
                tfidf_matrix = self.lsa.fit_transform(tfidf_matrix)

            document_index = 0
            query_vector = tfidf_matrix[document_index].reshape(1, -1)

            return cosine_similarity(query_vector, tfidf_matrix)[0][0]

        except Exception as e:
            # Print the exception and store the problematic documents
            print(f"Error processing documents: {e}")
            print("Doc1:", doc1)
            print("Doc2:", doc2)
            self.problematic_docs.append((doc1, doc2))
            return 0.0

    def binary_score(self, doc1, doc2):
        return float(int(self.score(doc1, doc2) > 0.5))

class Transformer:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def sentence_embeddings(self, model_output, encoded_input):
        # Perform pooling to compute sentence embedding
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def score(self, doc1, doc2):
        # Tokenize sentences
        # encoded_input = self.tokenizer([doc1, doc2], padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        # with torch.no_grad():
            # model_output = self.model(**encoded_input)

        # sentence_embeddings = self.sentence_embeddings(model_output, encoded_input)

        sentence_embeddings = self.model.encode([doc1, doc2])
        # Calculate cosine similarity
        return util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])[0][0]

    def fit(self, train_data, num_epochs):
        #Tune the model
        self.model.fit(train_objectives=[(DataLoader(train_data, shuffle=True, batch_size=16), losses.CosineSimilarityLoss(self.model))], epochs=num_epochs, warmup_steps=100)

class MpnetTransformer:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def sentence_embeddings(self, model_output, encoded_input):
        # Perform pooling to compute sentence embedding
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def encode(self, doc):
        return self.model.encode([doc])[0]

    def score(self, doc1, doc2):
        # Tokenize sentences
        # encoded_input = self.tokenizer([doc1, doc2], padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        # with torch.no_grad():
            # model_output = self.model(**encoded_input)

        # sentence_embeddings = self.sentence_embeddings(model_output, encoded_input)

        sentence_embeddings = self.model.encode([doc1, doc2])
        # Calculate cosine similarity
        return util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])[0][0]

    def binary_score(self, doc1, doc2):
        return float(int(self.score(doc1, doc2) > 0.5))

    def fit(self, train_data, num_epochs):
        #Tune the model
        self.model.fit(train_objectives=[(DataLoader(train_data, shuffle=True, batch_size=16), losses.CosineSimilarityLoss(self.model))], epochs=num_epochs, warmup_steps=100)

class DistilRobertaTransformer:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def sentence_embeddings(self, model_output, encoded_input):
        # Perform pooling to compute sentence embedding
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def encode(self, doc):
        return self.model.encode([doc])[0]

    def score(self, doc1, doc2):
        sentence_embeddings = self.model.encode([doc1, doc2])
        # Calculate cosine similarity
        return util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])[0][0]

    def binary_score(self, doc1, doc2):
        return float(int(self.score(doc1, doc2) > 0.5))

    def fit(self, train_data, num_epochs):
        #Tune the model
        self.model.fit(train_objectives=[(DataLoader(train_data, shuffle=True, batch_size=16), losses.CosineSimilarityLoss(self.model))], epochs=num_epochs, warmup_steps=100)

class SemanticSimilarityModel:
    def __init__(self, k = 1000):
        self.summerizer = Summerizer()
        self.lsa = LSA()
        self.mpnet = MpnetTransformer()
        self.distilroberta = DistilRobertaTransformer()
        self.k = k
        self.rf = RandomForestClassifier()

    def encode(self, doc):
        if len(doc) > self.k:
            doc = self.summerizer.summerize(doc)
        
        mpnet_embedding = self.mpnet.encode(doc)
        distilroberta_embedding = self.distilroberta.encode(doc)
        lsa_embedding = self.lsa.encode(doc)

        return [mpnet_embedding, distilroberta_embedding, lsa_embedding]

    def majority(self, scores):
        buckets = {}

        for score in scores:
            if not score in buckets:
                buckets[score] = 0
            buckets[score] += 1

        return max(buckets, key=buckets.get)

    def similarity_scores(self, doc1, doc2):
        if len(doc1) > self.k:
            doc1 = self.summerizer.summerize(doc1)

        if len(doc2) > self.k:
            doc2 = self.summerizer.summerize(doc2)


        mpnet_score = self.mpnet.binary_score(doc1, doc2)
        distilroberta_score = self.distilroberta.binary_score(doc1, doc2)
        lsa_score = self.lsa.binary_score(doc1, doc2)

        return [mpnet_score, distilroberta_score, lsa_score]
    
    def is_similar(self, doc1, doc2):
        scores = self.similarity_scores(doc1, doc2)
        
        return self.rf.predict([scores])[0]
        # return self.majority([mpnet_score, distilroberta_score, lsa_score])

    def evaluate(self, doc1, doc2, score):
        model_score = self.is_similar(doc1['text'], doc2['text'])

        lsa_score = self.lsa.score(doc1['text'], doc2['text'])

        mpnet_score = self.mpnet.score(doc1['text'], doc2['text'])

        distilroberta_score = self.distilroberta.score(doc1['text'], doc2['text'])

        return {
            "doc_id_1": str(doc1['id']),
            "doc_id_2": str(doc2['id']),

            "score": score,

            "model": model_score,
            "lsa": lsa_score,
            "mpnet": mpnet_score,
            "distilroberta": distilroberta_score,
            
            "model_loss": abs(model_score - score),
            "lsa_loss": abs(lsa_score - score),
            "mpnet_loss": abs(mpnet_score - score),
            "distilroberta_loss": abs(distilroberta_score - score),
        }

    def evaluate_batch(self, batch):
        result = []

        for item in tqdm(batch, total=len(batch), desc='Evaluating batch'):
            loss = self.evaluate(item['doc1'], item['doc2'], item['score'])
            result.append(loss)

        results_df = pd.DataFrame.from_records(result).astype('string')

        results_df['score'] = results_df['score'].astype('float')
        results_df['model'] = results_df['model'].astype('float')
        results_df['lsa'] = results_df['lsa'].astype('float')
        results_df['mpnet'] = results_df['mpnet'].astype('float')
        results_df['distilroberta'] = results_df['distilroberta'].astype('float')
        results_df['model_loss'] = results_df['model_loss'].astype('float')
        results_df['lsa_loss'] = results_df['lsa_loss'].astype('float')
        results_df['mpnet_loss'] = results_df['mpnet_loss'].astype('float')
        results_df['distilroberta_loss'] = results_df['distilroberta_loss'].astype('float')

        results_df['binary_score'] = np.where(results_df['score'] > 0.5, 1, 0)
        results_df['binary_model'] = np.where(results_df['model'] > 0.5, 1, 0)
        results_df['binary_lsa'] = np.where(results_df['lsa'] > 0.5, 1, 0)
        results_df['binary_mpnet'] = np.where(results_df['mpnet'] > 0.5, 1, 0)
        results_df['binary_distilroberta'] = np.where(results_df['distilroberta'] > 0.5, 1, 0)


        print('\n------------ ACCURACY ------------')
        print(f'ACC (Model): {accuracy_score(results_df["binary_score"], results_df["binary_model"])}')
        print(f'ACC (LSA): {accuracy_score(results_df["binary_score"], results_df["binary_lsa"])}')
        print(f'ACC (mpnet): {accuracy_score(results_df["binary_score"], results_df["binary_mpnet"])}')
        print(f'ACC (distilroberta): {accuracy_score(results_df["binary_score"], results_df["binary_distilroberta"])}')

        print('\n------------ F1 ------------')
        print(f'F1 (Model): {f1_score(results_df["binary_score"], results_df["binary_model"],zero_division=0)}')
        print(f'F1 (LSA): {f1_score(results_df["binary_score"], results_df["binary_lsa"],zero_division=0)}')
        print(f'F1 (mpnet): {f1_score(results_df["binary_score"], results_df["binary_mpnet"],zero_division=0)}')
        print(f'F1 (distilroberta): {f1_score(results_df["binary_score"], results_df["binary_distilroberta"],zero_division=0)}')

        print('\n------------ RMSE ------------')
        print(f'RMSE (Model): {np.sqrt(mean_squared_error(results_df["score"].values, results_df["model"].values))}')
        print(f'RMSE (LSA): {np.sqrt(mean_squared_error(results_df["score"].values, results_df["lsa"].values))}')
        print(f'RMSE (mpnet): {np.sqrt(mean_squared_error(results_df["score"].values, results_df["mpnet"].values))}')
        print(f'RMSE (distilroberta): {np.sqrt(mean_squared_error(results_df["score"].values, results_df["distilroberta"].values))}')

        print('\n------------ MAE ------------')
        print(f'MAE (Model): {results_df["model_loss"].mean()}')
        print(f'MAE (LSA): {results_df["lsa_loss"].mean()}')
        print(f'MAE (mpnet): {results_df["mpnet_loss"].mean()}')
        print(f'MAE (distilroberta): {results_df["distilroberta_loss"].mean()}')

        print('\n------------ MIN/MAX ------------')
        print(f'MIN LOSS (Model): {results_df.iloc[results_df["model_loss"].idxmin()]}')
        print(f'MAX LOSS (Model): {results_df.iloc[results_df["model_loss"].idxmax()]}')
        print(f'MIN LOSS (LSA): {results_df.iloc[results_df["lsa_loss"].idxmin()]}')
        print(f'MAX LOSS (LSA): {results_df.iloc[results_df["lsa_loss"].idxmax()]}')
        print(f'MIN LOSS (mpnet): {results_df.iloc[results_df["mpnet_loss"].idxmin()]}')
        print(f'MAX LOSS (mpnet): {results_df.iloc[results_df["mpnet_loss"].idxmax()]}')
        print(f'MIN LOSS (distilroberta): {results_df.iloc[results_df["distilroberta_loss"].idxmin()]}')
        print(f'MAX LOSS (distilroberta): {results_df.iloc[results_df["distilroberta_loss"].idxmax()]}')

    def transform(self, train_data):
        train_examples = []
        
        for d in tqdm(train_data, total=len(train_data), desc="Transforming data"):
            d1_text = d['doc1']['text']
            if len(d1_text) > self.k:
                d1_text = self.summerizer.summerize(d1_text)

            d2_text = d['doc2']['text']
            if len(d2_text) > self.k:
                d2_text = self.summerizer.summerize(d2_text)

            train_examples.append(InputExample(texts=[d1_text, d2_text], label=d['score']))

        return train_examples

    def fit_mpnet(self, train_data, num_epochs=3):
        print('fitting data')
        self.mpnet.fit(train_data, num_epochs)

    def fit_distilroberta(self, train_data, num_epochs=3):
        print('fitting data')
        self.distilroberta.fit(train_data, num_epochs)

    def fit_classifier(self, batch):
        print('fitting data')
        
        X = []
        y = []

        for item in tqdm(batch, total=len(batch), desc='Processing batch'):
            scores = self.similarity_scores(item['doc1']['text'], item['doc2']['text'])
            X.append(scores)
            y.append(int(item['score'] > 0.5))

        self.rf.fit(X, y)

    def save(self, fname):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        with open(f'{base}/snapshot/{fname}_mpnet.model', 'wb') as f:
            pickle.dump(self.mpnet, f)
        
        with open(f'{base}/snapshot/{fname}_distilroberta.model', 'wb') as f:
            pickle.dump(self.distilroberta, f)
    
        with open(f'{base}/snapshot/{fname}_rf.model', 'wb') as f:
            pickle.dump(self.rf, f)

    def load(self, fname):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        with open(f'{base}/snapshot/{fname}_mpnet.model', 'rb') as f:
            self.mpnet = pickle.load(f)
        
        with open(f'{base}/snapshot/{fname}_distilroberta.model', 'rb') as f:
            self.distilroberta = pickle.load(f)
        
        with open(f'{base}/snapshot/{fname}_rf.model', 'rb') as f:
            self.rf = pickle.load(f)
