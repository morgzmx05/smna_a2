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

def extract_comments_and_network(youtube, video_ids, max_comments_per_video=500):
    #Extracts comments from a manual list of videos and builds NLP & Network DataFrames.
    nlp_data = []
    network_edges = {} #Dictionary to track (Source, Target) to Weight
    
    for video_id in video_ids:
        print(f"Extracting data for video: {video_id}...")
        count = 0
        page_token = None
        
        while True:
            try:
                # Fetch Top Comment Threads in batches
                threads_response = youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, max_comments_per_video - count), #100 at a time
                    textFormat="plainText",
                    pageToken=page_token
                ).execute()
                
                for thread in threads_response.get("items", []):
                    top_comment = thread["snippet"]["topLevelComment"]["snippet"]
                    target_user = top_comment.get("authorDisplayName", "Unknown_User")
                    
                    # Append to NLP DataFrame list
                    nlp_data.append({
                        "Video_ID": video_id,
                        "User": target_user,
                        "Text": top_comment["textDisplay"],
                        "Date": top_comment["publishedAt"],
                        "Likes": top_comment["likeCount"],
                        "Is_Reply": False
                    })
                    count += 1
                    
                    # Fetch Replies to build the Network 
                    if thread["snippet"]["totalReplyCount"] > 0 and "replies" in thread:
                        for reply_item in thread["replies"]["comments"]: 
                            reply = reply_item["snippet"]
                            source_user = reply.get("authorDisplayName", "Unknown_User")
                            
                            # Append reply to NLP DataFrame list
                            nlp_data.append({
                                "Video_ID": video_id,
                                "User": source_user,
                                "Text": reply["textDisplay"],
                                "Date": reply["publishedAt"],
                                "Likes": reply["likeCount"],
                                "Is_Reply": True
                            })
                            count += 1
                            
                            # source_User replied to Target_User
                            if source_user != target_user: 
                                edge = (source_user, target_user)
                                network_edges[edge] = network_edges.get(edge, 0) + 1

                #Check if limit achieved or out of pages
                page_token = threads_response.get('nextPageToken')
                if not page_token or count >= max_comments_per_video:
                    print(f" Got ~{count} comments/replies.")
                    break
                    
            except Exception as e:
                print(f"ERROR: Failed or stopped for {video_id}: {e}")
                break
                
    #turn Network Data into list of dictionaries
    network_data = [{"Source_User": k[0], "Target_User": k[1], "Weight": v} for k, v in network_edges.items()]
    
    return pd.DataFrame(nlp_data), pd.DataFrame(network_data)

def get_video_metadata(youtube, video_ids):
    """Fetch video publication dates from YouTube API"""
    video_meta = {}
    for video_id in video_ids:
        try:
            response = youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()
            if response['items']:
                pub_date = response['items'][0]['snippet']['publishedAt']
                video_meta[video_id] = pd.to_datetime(pub_date)
        except Exception as e:
            print(f"Error fetching metadata for {video_id}: {e}")
    return video_meta

#extraction
if __name__ == "__main__":
    print(f"Starting extraction from manually chosen {len(vid_list)} videos")
    
    # Fetch video metadata first
    print("Fetching video publication dates...")
    video_metadata = get_video_metadata(youtube, vid_list)
    
    # increase max_comments_per_video for denser network
    df_nlp, df_network = extract_comments_and_network(youtube, vid_list, max_comments_per_video=1000)
    
    # Map video publication dates to dataframe
    df_nlp['Video_Published_Date'] = df_nlp['Video_ID'].map(video_metadata)
    
    # Save your datasets
    df_nlp.to_csv("housing_crisis_nlp_data.csv", index=False)
    df_network.to_csv("housing_crisis_network_data.csv", index=False)
    
    print("\nExtraction done")
    print(f"Total NLP Rows: {len(df_nlp)}")
    print(f"Total Network Edges: {len(df_network)}")
    
    # Video published range
    if len(df_nlp) > 0:
        # Convert dates to datetime format
        df_nlp['Date'] = pd.to_datetime(df_nlp['Date'])
        earliest_video = df_nlp['Video_Published_Date'].min()
        latest_video = df_nlp['Video_Published_Date'].max()
        avg_video_date = df_nlp[['Video_ID', 'Video_Published_Date']].drop_duplicates()['Video_Published_Date'].mean()
        earliest_comment = df_nlp['Date'].min()
        latest_comment = df_nlp['Date'].max()
        avg_comment_date = df_nlp['Date'].mean()
        print(f"\nVideo Publication Range:")
        print(f"Earliest video published: {earliest_video}")
        print(f"Latest video published: {latest_video}")
        print(f"Avg date video published: {avg_video_date}")
        print(f"\nComment Date Range:")
        print(f"Earliest comment: {earliest_comment}")
        print(f"Latest comment: {latest_comment}")
        print(f"Avg comment date: {avg_comment_date}")