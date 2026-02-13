# preprocessing/text_preprocessing.py

import re

# ===============================
# NEGATION WORDS (VERY IMPORTANT)
# ===============================
NEGATIONS = [
    "illa", "illai", "ille", "kedaiyathu",
    "not", "no", "never", "nothing",
    "mosam", "worst"
]

# ===============================
# THANGGLISH NORMALIZATION
# ===============================
THANGLISH_MAP = {
    "nalla": "good",
    "super": "good",
    "romba": "very",
    "pidikum": "like",
    "pidikkum": "like",
    "mosam": "bad",
    "illa": "not",
    "illai": "not",
    "ah": "",
    "da": "",
    "pa": ""
}

THANGLISH_MAP.update({
    "surprise": "surprise",
    "surprised": "surprise",
    "ayiten": "surprise",
    "aachariyam": "surprise",
    "aachariyama": "surprise"
})


# ===============================
# CLEAN TEXT FUNCTION
# ===============================
def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove emojis (optional)
    text = re.sub(r"[^\w\s\u0B80-\u0BFF]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # ===============================
    # THANGGLISH WORD NORMALIZATION
    # ===============================
    words = text.split()
    normalized_words = []

    for w in words:
        if w in THANGLISH_MAP:
            normalized_words.append(THANGLISH_MAP[w])
        else:
            normalized_words.append(w)

    text = " ".join(normalized_words)

    # ===============================
    # NEGATION HANDLING (CRITICAL)
    # ===============================
    words = text.split()
    final_words = []

    for i, word in enumerate(words):
        if word in NEGATIONS and i > 0:
            # attach negation to previous word
            final_words[-1] = "not_" + final_words[-1]
        else:
            final_words.append(word)

    text = " ".join(final_words)

    return text
