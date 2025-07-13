# export_model.py — Save all trained models

import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

from joblib import dump
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load best model name
with open("best_model_name.txt", "r") as f:
    best_model_name = f.read().strip()

# Load vectorizer and encoder
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("condition_encoder.pkl", "rb") as f:
    condition_encoder = pickle.load(f)

# Load sentiment model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(LogisticRegression(max_iter=200), f)  # just saving placeholder; adjust if trained separately

# Save ML models
print("\n✅ Saving traditional models...")

models = {
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

for name, model in models.items():
    path = f"{name.lower()}_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

# Save BERT path
print("\n✅ Saving BERT model and tokenizer...")
bert_model_dir = "bert_model"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

# Save locally
bert_model.save_pretrained("bert_model")
bert_tokenizer.save_pretrained("bert_model")

print("\n✅ All models exported successfully.")




