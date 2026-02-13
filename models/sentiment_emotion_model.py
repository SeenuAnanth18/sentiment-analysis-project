import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import os

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "xlm-roberta-base"

WEIGHTS_PATH = os.path.join(BASE_DIR, "sentiment_emotion_xlm_roberta.pth")
SENTIMENT_CLASSES_PATH = os.path.join(BASE_DIR, "sentiment_classes.pt")
EMOTION_CLASSES_PATH = os.path.join(BASE_DIR, "emotion_classes.pt")


# ================= LOAD MODEL =================
def load_model():

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # PyTorch 2.6 FIX
    sentiment_classes = torch.load(
        SENTIMENT_CLASSES_PATH,
        map_location="cpu",
        weights_only=False
    )

    emotion_classes = torch.load(
        EMOTION_CLASSES_PATH,
        map_location="cpu",
        weights_only=False
    )

    class XLMRSentimentEmotion(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = XLMRobertaModel.from_pretrained(MODEL_NAME)
            hidden = self.encoder.config.hidden_size

            self.sentiment_fc = nn.Linear(hidden, len(sentiment_classes))
            self.emotion_fc = nn.Linear(hidden, len(emotion_classes))

        def forward(self, input_ids, attention_mask):

            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            pooled = outputs.last_hidden_state[:, 0]

            sentiment_logits = self.sentiment_fc(pooled)
            emotion_logits = torch.sigmoid(self.emotion_fc(pooled))

            return sentiment_logits, emotion_logits

    model = XLMRSentimentEmotion()

    model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    )

    model.eval()

    return tokenizer, model, sentiment_classes, emotion_classes
