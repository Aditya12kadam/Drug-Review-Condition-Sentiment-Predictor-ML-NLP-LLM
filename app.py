
import streamlit as st
import pickle 
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.special import softmax
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from datetime import datetime
import random
sample_reviews = [
    "This medicine helped me sleep better after struggling with insomnia for weeks.",
    "I felt extremely nauseous and dizzy after taking just one pill. Never again!",
    "The medication worked wonders for my ADHD. I can finally focus during the day.",
    "I’ve had no side effects and my anxiety has reduced significantly.",
    "Didn't help with my arthritis pain and gave me a rash. Very disappointed.",
    "After 3 days of using it for migraines, my headaches are almost gone!",
    "Caused severe mood swings and fatigue. Had to stop taking it.",
    "Zoloft helped improve my depression symptoms within two weeks. Grateful for this drug.",
    "Lisinopril controlled my blood pressure well, but gave me a dry cough.",
    "Didn't notice much difference but no major side effects either."
]

st.set_page_config(layout="wide")
nltk.download('stopwords')
nltk.download('wordnet')


# Load models and encoders
@st.cache_resource
def load_sklearn_artifacts(): # chnge done 
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("condition_encoder.pkl", "rb") as f:
        condition_encoder = pickle.load(f)
    with open("sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    return vectorizer, condition_encoder, sentiment_model  #chnage
vectorizer, condition_encoder, sentiment_model = load_sklearn_artifacts() # change

@st.cache_resource  #
def load_sklearn_classifier_models():#
    models = {
        "PassiveAggressive": pickle.load(open("passiveaggressive_model.pkl", "rb")),
        "NaiveBayes": pickle.load(open("naivebayes_model.pkl", "rb")),
        "LogisticRegression": pickle.load(open("logisticregression_model.pkl", "rb")),
    }
    return models#
models = load_sklearn_classifier_models()#

@st.cache_resource #
def load_all_bert_models(): #
    #bert_tokenizer = BertTokenizer.from_pretrained("bert_model")
    #bert_model = BertForSequenceClassification.from_pretrained("bert_model")

    bert_tokenizer = AutoTokenizer.from_pretrained("bert_model")
    bert_model = AutoModelForSequenceClassification.from_pretrained("bert_model")

    #sentiment_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    #sentiment_bert = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment_bert = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return bert_tokenizer, bert_model, sentiment_tokenizer, sentiment_bert #
bert_tokenizer, bert_model, sentiment_tokenizer, sentiment_bert = load_all_bert_models()

@st.cache_resource
def load_llm():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device  # Ensures model and tensors are on the same device
    )
llm_pipe = load_llm()

stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

model_scores = {
    "BERT": 0.91,
    "PassiveAggressive": 0.8885,
    "NaiveBayes": 0.8420,
    "LogisticRegression": 0.8669,
}

model_scores_df = pd.DataFrame({
    'Model': list(model_scores.keys()),
    'Accuracy': list(model_scores.values())
})

if "history" not in st.session_state:
    st.session_state.history = []

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""


def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop and len(w) > 2])

def predict_condition(review, model_name):
    if model_name == "BERT":
        inputs = bert_tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        top3_indices = np.argsort(probs)[::-1][:3]
        class_labels = condition_encoder.classes_
        top3_labels = class_labels[top3_indices]
        top3_probs = probs[top3_indices]
        return list(zip(top3_labels, top3_probs)), review

    review_cleaned = clean_text(review)
    X_vec = vectorizer.transform([review_cleaned])
    model = models[model_name]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
    else:
        scores = model.decision_function(X_vec)
        probs = softmax(scores) if scores.ndim == 1 else softmax(scores[0])

    top3_indices = np.argsort(probs)[::-1][:3]
    class_labels = condition_encoder.classes_
    top3_labels = class_labels[top3_indices]
    top3_probs = np.array(probs)[top3_indices]
    return list(zip(top3_labels, top3_probs)), review_cleaned

def predict_sentiment_bert(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = sentiment_bert(**inputs)
        #scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        sentiment = np.argmax(scores) + 1
    return "Positive" if sentiment >= 4 else "Neutral" if sentiment == 3 else "Negative"

def explain_with_llm(review):
    prompt = ( f"Based on the following medical review, provide a brief explanation of the likely condition and sentiment:\nReview: {review}\nExplanation:")
    try:
        result = llm_pipe(
            prompt,
            max_new_tokens=120,  # Increase for longer explanation
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )[0]["generated_text"]

        # Remove repeated prompt if any and trim unnecessary text
        explanation = result.split("Explanation:")[-1].strip()
        return explanation.replace("\n", " ").strip()
    except Exception as e:
        return f"⚠️ LLM explanation could not be generated. Reason: {str(e)}"


def display_wordcloud(text):
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# UI Starts
st.title("💊 Drug Review Condition & Sentiment Predictor")
st.markdown("Made by **Aditya Kadam** | NLP + ML + BERT + TinyLLaMA")



with st.sidebar:
    st.header("🔍 About")
    st.markdown("This app predicts the patient's condition and sentiment from a drug review.")
    model_choice = st.selectbox("Choose Condition Model:", ["BERT"] + list(models.keys()), help="Select model to classify condition")

    #st.markdown("---")
    # st.subheader("📊 Model Accuracy Comparison")
    #st.bar_chart(model_scores)

user_input = st.text_area("📝 Enter Drug Review:", value=st.session_state.user_input if "user_input" in st.session_state else "", height=150, key="input_box")

cols = st.columns([4, 1])  # Predict wider, Sample narrower


with cols[0]:
    predict_clicked = st.button("🔮 Predict", key="predict_button")

with cols[1]:
    if st.button("🧪 Sample Inputs", key="sample_button"):
        st.session_state.user_input = random.choice(sample_reviews)
        st.rerun()


predictions, cleaned = None, None

if predict_clicked:
   # user_input = user_input  # Already assigned above

    if len(user_input.split()) < 5:

        st.warning("⚠️ Please enter a more descriptive review (at least 5 words).")
    elif user_input.strip():
        predictions, cleaned = predict_condition(user_input, model_choice)
        sentiment = predict_sentiment_bert(user_input)
        explanation = explain_with_llm(user_input)

        st.session_state.history.append({
            "Review": user_input,
            "Model": model_choice,
            "Top Condition": predictions[0][0] if predictions else None,
            "Confidence": f"{predictions[0][1]:.2f}" if predictions else None,
            "Sentiment": sentiment,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        tabs = st.tabs(["📋 Predictions", "☁️ Word Cloud", "📈 Analytics","📊 Model Comparison"])

        with tabs[0]:
            st.subheader("📋 Top 3 Predicted Conditions:")
            for label, prob in predictions:
                st.write(f"🔹 {label.strip().title()} — {prob:.4f}")

            st.subheader("💬 Sentiment:")
            st.success(sentiment)

            st.subheader("🧠 Explanation (LLM)")
            st.info(explanation)

        with tabs[1]:
            if cleaned:
                display_wordcloud(cleaned)

        with tabs[2]:
            if cleaned:
                st.markdown(f"- Total words: {len(cleaned.split())}")
                st.markdown(f"- Vocabulary size: {len(vectorizer.get_feature_names_out())}")
                st.markdown(f"- Conditions in dataset: {len(condition_encoder.classes_)}")

           
            st.subheader("📈 Prediction Confidence Chart")
            probs_df = {label.strip().title(): float(prob) for label, prob in predictions}
            st.bar_chart(probs_df)

        with tabs[3]: # <--- NEW TAB FOR MODEL COMPARISON
            st.subheader("📊 Model Accuracy Comparison")
            st.bar_chart(model_scores_df, x='Model', y='Accuracy')
        

if st.session_state.history:
    st.markdown("---")
    st.subheader("🕘 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, use_container_width=True)
    st.download_button("Download History as CSV", hist_df.to_csv(index=False), file_name="prediction_history.csv")
