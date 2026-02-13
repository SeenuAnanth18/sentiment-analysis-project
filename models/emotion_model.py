# models/emotion_model.py

import torch
from models.sentiment_emotion_model import load_model

tokenizer, model, _, emotion_classes = load_model()


def detect_emotion(text, threshold=0.25):
    """
    Multilingual emotion detection
    Supports Tamil + English + Thanglish
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        _, emo_logits = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

    probs = emo_logits.squeeze().tolist()

    emotions = [
        emotion_classes[i]
        for i, p in enumerate(probs)
        if p >= threshold
    ]

    # choose highest emotion if none crosses threshold
    if not emotions:
        max_idx = probs.index(max(probs))
        emotions = [emotion_classes[max_idx]]

    return emotions
