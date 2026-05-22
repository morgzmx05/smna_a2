import pandas as pd
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()

# API, Replace with YouTube Data API v3 Key
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Manually picked video IDs (add if more found)
vid_list = [
    "vWlh41jJ2Ww", 
    "IqKIlLS7ln4",
    "PQn41fzVry0", 
    "01BDbhKm274",
    "JqSbECqD7bc",
    "4IihsPhVd7c",
    "RfzDHc5dF9Q",
    "yNH91o9hP8w",
    "glVqpSLyrZc",
    "feffoaDAgO0",
    "a8iNcHeBUCU",
    "qHAXbR7SYP8",
    "H-o_F6nKOro",
    "4IihsPhVd7c",
    "g29yvU-6ep4",
    "PNRl8TsMa9w",
    "c37eEtRLPjU",
    "HRP1t-P0Ljw",
    "w2XCq5VyITI",
    "h3Ny2J6eNsM",
    "hk9CbqKEl-w",
    "u71XD1Bt6Wo",
    "LuEaGNXlh1E",
    "2B4wf5E44bw",
    "kOJ8PoOTFhI",
    "uUSLmYPHkCc",
    "GAu1Y-5EPeM",
    "jV6BprD7toY",
    "vSfec19YOPg",
    "x6e1PGPdepM",
    "JMin2Czkx1M",
    "RJDVeLa7uXg"
]

def get_video_metadata(youtube, video_ids):
    """Fetches title and publishing date for a list of video IDs."""
    metadata = []
    
    # YouTube API accepts up to 50 IDs per request
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        
        try:
            response = youtube.videos().list(
                part="snippet",
                id=",".join(batch)
            ).execute()
            
            for item in response.get("items", []):
                metadata.append({
                    "Video_ID": item["id"],
                    "Title": item["snippet"]["title"],
                    "Published_At": item["snippet"]["publishedAt"],
                    "Channel": item["snippet"]["channelTitle"]
                })
                
        except Exception as e:
            print(f"ERROR fetching metadata for batch {i}: {e}")
    
    return pd.DataFrame(metadata)


if __name__ == "__main__":
    # existing extraction code...

    # new metadata extraction
    df_metadata = get_video_metadata(youtube, vid_list)
    df_metadata.to_csv("housing_crisis_video_metadata.csv", index=False)
    print(f"\nVideo metadata saved: {len(df_metadata)} videos")
    print(df_metadata[["Video_ID", "Title", "Published_At"]].to_string())