# Australian Housing Crisis: YouTube Comment Network Analysis
**RMIT University - Social Media and Network Analysis (COSC 2671) Assignment 2**

## Project Overview
This project investigates the social dynamics and echo chambers present in YouTube comment sections regarding the Australian housing crisis. Specifically, we analyse discussions surrounding polarising topics such as negative gearing, immigration, and rent. 

Our core research questions are:
1. **Community Detection:** Are there distinct user communities (echo chambers) forming around specific ideologies?
2. **Influence Propagation:** Do central, highly-replied-to users drive specific narrative topics within the network?
3. **Sentiment Polarisation:** Do different topological communities exhibit contrasting sentiments?

## Data Source & Collection
Data was collected using a custom Python script interfacing with the **YouTube Data API v3**. 

* **Source:** YouTube comment threads and replies.
* **Selection:** A curated list of highly controversial videos discussing the Australian housing crisis, published between May 2025 and May 2026.
* **Extraction Method:** The script (`housing_network.py`) extracts up to 500 comments/replies per video. 
* **Limitations:** The YouTube API inherently caps nested replies returned in the standard `commentThreads` endpoint to 5 per top-level comment. Additionally, some major news networks disable comments on highly controversial videos, requiring manual selection of the video IDs. 

### Network Construction
* **Nodes:** YouTube Users (`authorDisplayName`).
* **Edges:** Directed and Weighted. An edge represents `Source_User` (the person replying) interacting with `Target_User` (the original commenter).
* **Weight:** The frequency of replies from the `Source_User` to the `Target_User`. Self-replies are filtered out to maintain interaction integrity.

## Repository Structure
* `housing_network.py`: The data extraction and network-building script.
* `requirements.txt`: Python package dependencies.
* `.gitignore`: Prevents heavy and sensitive files like API keys from being committed.
* `/data/`:
  * `housing_crisis_nlp_data.csv`: Contains User, Text, Date, Likes, and Is_Reply flags for text/sentiment analysis.
  * `housing_crisis_network_data.csv`: Contains Source_User, Target_User, and Weight for graph modeling.

## Setup & Installation
This project requires Python 3.x. 

**1. Clone the repository**
`git clone <your-repository-url>`
`cd <your-repository-folder>`

**2. Install Dependencies**
Ensure you are running the exact same environment as the rest of the team:
`pip install -r requirements.txt`

**3. API Key Setup (Crucial)**
Do **NOT** hardcode your YouTube API key into the script. To run this project, you must securely configure your own key:
* Create a new file named exactly `.env` in the root folder of this project (the same folder as `housing_network.py`).
* Open the `.env` file and add your Google Cloud YouTube Data API v3 key exactly shown below, with no quotes or spaces:
`YOUTUBE_API_KEY=your_api_key_goes_here`
*(Note: The `.env` file is listed in our `.gitignore` and will never be uploaded to the repository, keeping your key safe on your local machine).*

**4. Run the Extraction**
Once your environment is set up and your key is in place, you can run the data extraction pipeline:
`python housing_network.py`