from googleapiclient.discovery import build
from config.config import API_KEY

def get_comments(video_id, max_results=50):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        text = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(text)

    return comments
