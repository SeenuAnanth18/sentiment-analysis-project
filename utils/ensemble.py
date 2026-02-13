from collections import Counter

def ensemble_voting(results):
    """
    results order:
    [TextBlob, VADER, BERT, XLM-RoBERTa]
    """

    weights = {
        0: 1,   # TextBlob
        1: 1,   # VADER
        2: 2,   # BERT multilingual
        3: 5    # YOUR TRAINED MODEL ⭐ MAIN MODEL
    }

    score = {}

    for i, r in enumerate(results):
        score[r] = score.get(r, 0) + weights[i]

    return max(score, key=score.get)
