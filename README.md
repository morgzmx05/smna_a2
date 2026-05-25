# Australian Housing Crisis: YouTube Comment Network Analysis
**RMIT University - Social Media and Network Analysis (COSC 2671) Assignment 2**

## Project Overview
This project investigates narrative polarization and echo chambers in YouTube comment discussions regarding the Australian housing crisis. We employ a **bipartite network projection** to analyse how video audiences overlap and form distinct thematic communities, then correlate these with sentiment and emotional patterns.

### Core Research Questions:
1. **Narrative Communities:** Do videos naturally cluster into distinct ideological/thematic groups based on shared audiences?
2. **Audience Polarisation:** Are there isolated narrative clusters with minimal audience overlap (echo chambers)?
3. **Sentiment Alignment:** Do different video communities exhibit contrasting emotions in their comment discussions?

## Data Source & Collection
Data was collected using a custom Python script interfacing with the **YouTube Data API v3**. 

* **Source:** YouTube comment threads and replies.
* **Selection:** A curated list of highly controversial videos discussing the Australian housing crisis, published between June 2023 and May 2026, with the average publish date being June 2025.
* **Extraction Method:** The script (`housing_network.py`) extracts up to 1000 comments/replies per video. 
* **Limitations:** The YouTube API inherently caps nested replies returned in the standard `commentThreads` endpoint to 5 per top-level comment. Additionally, some major news networks disable comments on highly controversial videos, requiring manual selection of the video IDs. 

### Data snapshot (example results)

- Bipartite network and projection statistics are produced by the analysis pipeline and recorded in the CSV outputs in the `outputs/` or project root (see **Data Outputs**).
- The video projection uses a configurable threshold to create weighted edges from shared commenters (default set in `network_analysis.py`).
- Community detection is performed with the Louvain algorithm by default; numbers (community counts, sizes, densities) will vary between runs and depend on the projection threshold and input data.

Note: Exact counts shown in past reports are preserved in the generated CSVs; re-run the pipeline to reproduce or update them for your dataset.

## Repository Structure

* `housing_network.py`: Data extraction and network-building script (YouTube API wrapper).
* `getVideoTitleAndInfo.py`: Helper script to fetch video metadata (titles, publish dates).
* `textProcessing.py`: Text preprocessing utilities used before topic and emotion analysis.
* `nlp_analysis.py`: NLP pipeline (BERTopic topic modelling + RoBERTa emotion classification).
* `topic_labels.py`: Utilities to load and apply human-readable topic labels.
* `network_analysis.py`: Video network construction, projection and community detection.
* `community_integration.py`: Community-level analysis (topic-emotion relations, statistics).
* `graph_visualisation.py`: Generates report visualisations and exports PNGs.
* `ldaVis.html`: Interactive topic visualisation produced by BERTopic/LDA exports.
* `TextAnalysisGraphs.ipynb`: Notebook for extra sentiment and topic graphs, text analysis, and interactive analysis.
* `requirements.txt`: Python package dependencies.
* `.gitignore`: Prevents heavy and sensitive files from being committed.

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

**5. Run the NLP Analysis**
Process raw comments using BERTopic (topic modeling) and RoBERTa (emotion classification):
`python nlp_analysis.py`

**6. Run the Network Analysis**
Analyse video communities via bipartite projection and Louvain community detection:
`python network_analysis.py`

This generates:
- `video_network_metrics.csv` - Per-video metrics (degree, betweenness, community)
- `video_community_analysis.csv` - Community cohesion, shared audiences, topic profiles
- `video_network_statistics.csv` - Network-wide statistics (modularity, density)

**7. Video Info**
Run `getVideoTitleandInfo.py ` to grab metadata used for graph visualisation

**8. Topic labels**
Run `topic_labels.py to generate BERTopic dictionary .csv file used for graph visualisation

**9 Community integration**
Run `community_integration.py` to generate 
.csv files used for graphs and analysis

**10. Visualisation**
Run `graph_visualisation.py` for graphs related to the following:
* video network
* emotion distribution
* topic distribution
* centrality analysis
* echo chamber coherence
* community scale
* top creators
* summary statistics table

**11. Open the Jupyter Notebook**
Run the interactive notebook for exploratory analysis and ad-hoc graphs:

```bash
# start Jupyter in the project root, then open `TextAnalysisGraphs.ipynb`
jupyter lab
# or
jupyter notebook
```
The notebook contains exploratory plots and examples of how to visualise and export figures (PNG/SVG) from the analysis outputs.

Note: Important to clear all outputs and restart before running.

## **Data Outputs**
* `housing_crisis_nlp_data.csv`: Raw comment text and metadata extracted via YouTube API.
* `housing_crisis_network_data.csv`: Bipartite edge list (user, video) and/or projected video edge list (source, target, weight).
* `housing_crisis_nlp_analysed.csv`: Processed NLP output including:
    * `Topic_Cluster` — cluster IDs from BERTopic
    * `Predominant_Emotion` — emotion label from RoBERTa
* `topic_dictionary.csv`: Topic summary and example comment previews (produced by `topic_labels.py`).
* `housing_crisis_video_metadata.csv`: Video metadata (Video_ID, Title, Published_At, Channel) produced by `getVideoTitleAndInfo.py`.
* `video_network_metrics.csv`: Per-video metrics (degree, betweenness, assigned community)
* `video_community_analysis.csv`: Aggregated community-level summaries (cohesion, audience size, topic profiles)
* Community integration outputs (from `community_integration.py`):
  - `community_emotion_matrix.csv` - emotion % by community
  - `community_topic_matrix.csv` - topic % by community
  - `community_sentiment_integration_summary.csv` - per-community summary statistics
  - `community_topic_emotion_coupling.csv` - topic-emotion associations per community
* `vis_*.png` and `ldaVis.html`: Generated visual assets and interactive topic visualisation for the report.
* `TextAnalysisGraphs.ipynb`: Notebook file used for exploratory analysis and optional figure exports (PNG/SVG).