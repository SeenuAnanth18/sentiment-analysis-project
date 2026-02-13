# ========================================
# 1. IMPORT LIBRARIES
# ========================================
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# ========================================
# 2. LOAD DATASET
# ========================================
df = pd.read_csv("Dataset.csv")

df.rename(columns={"comment": "comment_text"}, inplace=True)
df.dropna(inplace=True)

# ========================================
# 3. SENTIMENT ENCODING
# ========================================
sentiment_encoder = LabelEncoder()
df["sentiment_label"] = sentiment_encoder.fit_transform(df["sentiment"])

torch.save(sentiment_encoder.classes_, "sentiment_classes.pt")

# ========================================
# 4. EMOTION ENCODING (MULTI-LABEL)
# ========================================
df["emotion"] = df["emotion"].apply(lambda x: x.split("|"))

emotion_encoder = MultiLabelBinarizer()
emotion_labels = emotion_encoder.fit_transform(df["emotion"])

torch.save(emotion_encoder.classes_, "emotion_classes.pt")

# ========================================
# 5. TOKENIZER (XLM-RoBERTa)
# ========================================
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ========================================
# 6. DATASET CLASS
# ========================================
class CommentDataset(Dataset):
    def __init__(self, texts, sentiments, emotions):
        self.texts = texts
        self.sentiments = sentiments
        self.emotions = emotions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "sentiment": torch.tensor(self.sentiments[idx], dtype=torch.long),
            "emotion": torch.tensor(self.emotions[idx], dtype=torch.float)
        }

# ========================================
# 7. TRAIN / TEST SPLIT
# ========================================
X_train, X_test, y_sent_train, y_sent_test, y_emo_train, y_emo_test = train_test_split(
    df["comment_text"].tolist(),
    df["sentiment_label"].tolist(),
    emotion_labels,
    test_size=0.2,
    random_state=42
)

train_dataset = CommentDataset(X_train, y_sent_train, y_emo_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ========================================
# 8. MODEL (XLM-RoBERTa)
# ========================================
class SentimentEmotionModel(nn.Module):
    def __init__(self, emotion_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        self.sentiment_fc = nn.Linear(hidden_size, 3)
        self.emotion_fc = nn.Linear(hidden_size, emotion_classes)

    def forward(self, ids, mask):
        outputs = self.encoder(input_ids=ids, attention_mask=mask)
        x = outputs.last_hidden_state[:, 0, :]

        sentiment_out = self.sentiment_fc(x)
        emotion_out = torch.sigmoid(self.emotion_fc(x))

        return sentiment_out, emotion_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentEmotionModel(emotion_labels.shape[1]).to(device)

# ========================================
# 9. LOSS & OPTIMIZER
# ========================================
sentiment_loss_fn = nn.CrossEntropyLoss()
emotion_loss_fn = nn.BCELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ========================================
# 10. TRAINING LOOP
# ========================================
model.train()
for epoch in range(3):
    total_loss = 0

    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        sent = batch["sentiment"].to(device)
        emo = batch["emotion"].to(device)

        optimizer.zero_grad()
        sent_out, emo_out = model(ids, mask)

        loss = sentiment_loss_fn(sent_out, sent) + emotion_loss_fn(emo_out, emo)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# ========================================
# 11. SAVE MODEL
# ========================================
torch.save(model.state_dict(), "sentiment_emotion_xlm_roberta.pth")
print("✅ XLM-RoBERTa model trained & saved successfully")
