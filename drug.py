# drug_code.py — Train condition models without running BERT again

run_all = True  # Enable ML models but skip BERT below
run_bert = False  # NEW flag to control BERT execution

import pandas as pd
import re
import pickle
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("drugsComTrain.tsv", sep="\t")
df = df[["condition", "review", "rating"]].dropna()

# Keep top 20 conditions
top_conditions = df["condition"].value_counts().head(20).index
df = df[df["condition"].isin(top_conditions)]

# Add sentiment
def get_sentiment(r):
    return "Positive" if r >= 7 else "Negative" if r <= 4 else "Neutral"

df["sentiment"] = df["rating"].apply(get_sentiment)

# Clean text
stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop and len(w) > 2])

df["review_clean"] = df["review"].astype(str).apply(clean_text)

# Encode + vectorize
X = df["review_clean"]
y_condition = df["condition"]
y_sentiment = df["sentiment"]

condition_encoder = LabelEncoder()
y_condition_enc = condition_encoder.fit_transform(y_condition)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=5, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# Always train sentiment model
print("\n🔁 Training sentiment model only...")
X_sent_train, _, y_sent_train, _ = train_test_split(
    X_vec, y_sentiment, test_size=0.2, stratify=y_sentiment, random_state=42
)
sentiment_model = LogisticRegression(max_iter=200)
sentiment_model.fit(X_sent_train, y_sent_train)

# Save sentiment model + tools
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("condition_encoder.pkl", "wb") as f:
    pickle.dump(condition_encoder, f)

print("✅ Sentiment model saved.")

# If run_all, train ML models for condition
if run_all:
    print("\n⚙️ Training condition classifiers (ML models)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_condition_enc, test_size=0.2, stratify=y_condition_enc, random_state=42
    )
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = {
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000),
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=500)
    }

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        with open(f"{name.lower()}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ {name} Accuracy: {acc:.4f}")

# Skip BERT unless run_bert = True
if run_all and run_bert:
    print("\n🚀 Fine-tuning BERT... (skipped)")
