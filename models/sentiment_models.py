from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import torch.nn.functional as F

from models.sentiment_emotion_model import load_model

# load trained model once
tokenizer, model, sentiment_classes, _ = load_model()

vader = SentimentIntensityAnalyzer()

bert = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# ---------- TextBlob ----------
def textblob_sentiment(text):
    p = TextBlob(text).sentiment.polarity
    return "Positive" if p > 0 else "Negative" if p < 0 else "Neutral"


# ---------- VADER ----------
def vader_sentiment(text):
    s = vader.polarity_scores(text)["compound"]
    return "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"


# ---------- BERT ----------
def bert_sentiment(text):
    label = bert(text)[0]["label"]
    if label in ["4 stars", "5 stars"]:
        return "Positive"
    elif label in ["1 star", "2 stars"]:
        return "Negative"
    else:
        return "Neutral"


# ---------- YOUR TRAINED XLM-R ----------
def roberta_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits, _ = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

    probs = torch.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()

    return sentiment_classes[idx]

