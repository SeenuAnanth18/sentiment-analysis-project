# =====================================================
# 1. IMPORTS
# =====================================================
import pandas as pd
import re
from googleapiclient.discovery import build

# =====================================================
# 2. YOUTUBE API SETUP
# =====================================================
API_KEY = "AIzaSyAnELffornbJ-5SEfqZV0jkevGdloL-7sk"
youtube = build("youtube", "v3", developerKey=API_KEY)

# =====================================================
# 3. EXTRACT VIDEO ID
# =====================================================
def extract_video_id(url):
    return url.split("v=")[-1].split("&")[0]

# =====================================================
# 4. TEXT NORMALIZATION
# =====================================================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # reduce repeated letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# 5. FULL SPAM + IRRELEVANT FILTER
# =====================================================
def is_valid_comment(text):

    text_lower = text.lower()

    # Remove short comments
    if len(text_lower.split()) < 5:
        return False

    # Spam keywords
    spam_keywords = [
        "first", "subscribe", "subscribed",
        "who watching", "anyone here",
        "check my channel", "giveaway",
        "free", "click here"
    ]
    if any(keyword in text_lower for keyword in spam_keywords):
        return False

    # Budget / price suggestion comments
    suggest_keywords = [
        "suggest", "recommend",
        "best phone under", "best mobile under",
        "under 30k", "under 20k", "under 25k", "under 50k",
        "below 30000", "below 20000",
        "budget phone", "budget mobile",
        "which one to buy", "confused between",
        "laptop under", "price range"
    ]
    if any(keyword in text_lower for keyword in suggest_keywords):
        return False

    # Remove dynamic price patterns (25k, 30000, under 35000)
    if re.search(r"\b\d{2,5}\s?k\b", text_lower):
        return False
    if re.search(r"\bunder\s?\d+", text_lower):
        return False
    if re.search(r"\bbelow\s?\d+", text_lower):
        return False

    # Review request comments
    review_request_keywords = [
        "review podunga", "review pannunga",
        "next review", "please review",
        "waiting for review", "do review",
        "make review", "upload review",
        "unboxing podunga", "next video","compare","comparison","ads","advertisement"
    ]
    if any(keyword in text_lower for keyword in review_request_keywords):
        return False

    return True

# =====================================================
# 6. FETCH COMMENTS
# =====================================================
def fetch_comments(video_id, limit=1000):
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < limit:
        response = request.execute()

        for item in response["items"]:
            raw_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            clean_text = normalize_text(raw_text)

            if is_valid_comment(clean_text):
                comments.append(clean_text)

        request = youtube.commentThreads().list_next(request, response)

    return comments

# =====================================================
# 7. VIDEO LINKS (Mobile & Laptop Reviews)
# =====================================================
VIDEO_LINKS = [
    "https://www.youtube.com/watch?v=j3Ak1TQLFEg",
    "https://www.youtube.com/watch?v=7QY8xt2Qxmk",
    "https://www.youtube.com/watch?v=NHlLPTRM8JI",
    "https://www.youtube.com/watch?v=7wiY1id47i0",
    "https://www.youtube.com/watch?v=xpymilXdsEQ",
    "https://www.youtube.com/watch?v=I-05lVYB81k",
    "https://www.youtube.com/watch?v=LhJ0hYzsUZs"
]

all_comments = []

for link in VIDEO_LINKS:
    video_id = extract_video_id(link)
    comments = fetch_comments(video_id, limit=1200)
    all_comments.extend(comments)

# Remove duplicates
all_comments = list(set(all_comments))

# Keep best 2000
all_comments = all_comments[:2000]

# =====================================================
# 8. SAVE DATASET
# =====================================================
df = pd.DataFrame({"comment_text": all_comments})
df.to_csv("mobile_laptop_reviews_clean_2000.csv", index=False)

print("✅ High-quality review dataset created")
print("Total comments:", len(df))
print(df.head())
