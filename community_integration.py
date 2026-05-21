import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway, kruskal
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

nlp_df = pd.read_csv('housing_crisis_nlp_analysed.csv')
video_metrics = pd.read_csv('video_network_metrics.csv')

print(f"NLP data: {len(nlp_df)} comments")
print(f"Video metrics: {len(video_metrics)} videos")

communities = {
    0: ['JMin2Czkx1M', 'uUSLmYPHkCc', 'jV6BprD7toY', 'c37eEtRLPjU', 'qHAXbR7SYP8', 'GAu1Y-5EPeM', 'PNRl8TsMa9w', 'LuEaGNXlh1E'],
    1: ['4IihsPhVd7c', 'H-o_F6nKOro', 'HRP1t-P0Ljw', 'a8iNcHeBUCU', 'feffoaDAgO0', 'g29yvU-6ep4', 'vSfec19YOPg', 'h3Ny2J6eNsM', 'glVqpSLyrZc', 'x6e1PGPdepM', 'RJDVeLa7uXg', 'yNH91o9hP8w', 'RfzDHc5dF9Q'],
    2: ['2B4wf5E44bw', 'JqSbECqD7bc', 'u71XD1Bt6Wo', 'PQn41fzVry0', 'w2XCq5VyITI', 'IqKIlLS7ln4', 'kOJ8PoOTFhI', 'vWlh41jJ2Ww', 'hk9CbqKEl-w']
}

video_to_community = {}
for comm_id, videos in communities.items():
    for video in videos:
        video_to_community[video] = comm_id

nlp_df['Video_Community'] = nlp_df['Video_ID'].map(video_to_community)

# Remove rows with unmapped vids
nlp_df = nlp_df.dropna(subset=['Video_Community'])
nlp_df['Video_Community'] = nlp_df['Video_Community'].astype(int)

print(f"\nComments mapped to communities:")
for comm_id in sorted(nlp_df['Video_Community'].unique()):
    count = len(nlp_df[nlp_df['Video_Community'] == comm_id])
    print(f"  Community {comm_id}: {count} comments")

# Emotion contingency table
emotion_contingency = pd.crosstab(nlp_df['Video_Community'], nlp_df['Predominant_Emotion'])
chi2, p_val, dof, expected = chi2_contingency(emotion_contingency)

print(f"\nChi-Square Test for Independence:")
print(f"  H0: Emotions are independent of community")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_val:.4e}")
print(f"  Degrees of freedom: {dof}")
print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")

if p_val < 0.05:
    print(f"  Communities have statisitically different emotional profiles")
else:
    print(f"  No significant emotional difference between communities")

# Get normalized emotion distributions for each community
emotion_distributions = {}
for comm_id in sorted(nlp_df['Video_Community'].unique()):
    comm_emotions = nlp_df[nlp_df['Video_Community'] == comm_id]['Predominant_Emotion'].value_counts(normalize=True)
    all_emotions = nlp_df['Predominant_Emotion'].unique()
    dist = np.array([comm_emotions.get(e, 0) for e in all_emotions])
    emotion_distributions[comm_id] = dist

# Calculate pairwise Jensen-Shannon distances
print("\nEmotional Distance Matrix:")
print("(Higher = more different emotional profiles)")
from itertools import combinations

distances = {}
for comm_a, comm_b in combinations(sorted(emotion_distributions.keys()), 2):
    js_dist = jensenshannon(emotion_distributions[comm_a], emotion_distributions[comm_b])
    distances[(comm_a, comm_b)] = js_dist
    print(f"  Community {comm_a} vs Community {comm_b}: {js_dist:.4f}")

# emotions by community
emotion_by_community = pd.crosstab(nlp_df['Video_Community'], nlp_df['Predominant_Emotion'], normalize='index') * 100

for comm_id in sorted(emotion_by_community.index):
    if isinstance(comm_id, (int, float)) and not pd.isna(comm_id):
        print(f"\nCommunity {int(comm_id)}:")
        top_emotions = emotion_by_community.loc[comm_id].nlargest(8)
        for emotion, pct in top_emotions.items():
            print(f"  {emotion:15s}: {pct:5.1f}%")

# Topic by community
topic_by_community = nlp_df.groupby('Video_Community')['Topic_Cluster'].value_counts(normalize=True).groupby(level=0).head(6) * 100

for comm_id in sorted(nlp_df['Video_Community'].unique()):
    if isinstance(comm_id, (int, float)) and not pd.isna(comm_id):
        comm_topics = nlp_df[nlp_df['Video_Community'] == comm_id]['Topic_Cluster'].value_counts(normalize=True) * 100
        print(f"\nCommunity {int(comm_id)}:")
        for topic, pct in comm_topics.head(6).items():
            print(f"  Topic {int(topic):2d}: {pct:5.1f}%")

# Which topics drive which emotions
for comm_id in sorted(nlp_df['Video_Community'].unique()):
    if isinstance(comm_id, (int, float)) and not pd.isna(comm_id):
        comm_data = nlp_df[nlp_df['Video_Community'] == comm_id]
        
        # Top 3 topics
        top_topics = comm_data['Topic_Cluster'].value_counts().head(3).index
        
        print(f"Community {int(comm_id)} - Topic-Emotion Associations:")
        for topic in top_topics:
            topic_data = comm_data[comm_data['Topic_Cluster'] == topic]
            if len(topic_data) > 0:
                top_emotion = topic_data['Predominant_Emotion'].value_counts().index[0]
                emotion_pct = 100 * topic_data['Predominant_Emotion'].value_counts().iloc[0] / len(topic_data)
                print(f"  Topic {int(topic):2d} → {top_emotion:15s} ({emotion_pct:.1f}%)")
        print()

# Central vids analysis
# Merge video metrics with NLP data
video_metrics['Video_Community'] = video_metrics['Video_ID'].map(video_to_community)
video_metrics = video_metrics.dropna(subset=['Video_Community'])
video_metrics['Video_Community'] = video_metrics['Video_Community'].astype(int)

# Categorize vids by centrality
video_metrics['Centrality_Category'] = pd.cut(
    video_metrics['Betweenness_Centrality'],
    bins=[0, video_metrics['Betweenness_Centrality'].quantile(0.33), 
          video_metrics['Betweenness_Centrality'].quantile(0.66), 1.0],
    labels=['Low', 'Medium', 'High']
)

for category in ['Low', 'Medium', 'High']:
    high_centrality_videos = video_metrics[video_metrics['Centrality_Category'] == category]['Video_ID'].tolist()
    category_data = nlp_df[nlp_df['Video_ID'].isin(high_centrality_videos)]
    
    if len(category_data) > 0:
        top_emotion = category_data['Predominant_Emotion'].value_counts().index[0]
        neutral_pct = 100 * (category_data['Predominant_Emotion'] == 'neutral').sum() / len(category_data)
        print(f"\n{category:6s} Centrality ({len(high_centrality_videos):2d} videos, {len(category_data):5,} comments):")
        print(f"  Dominant emotion: {top_emotion:15s} | Neutral: {neutral_pct:.1f}%")

# Load community cohesion data
community_cohesion = {
    0: 0.8929,  # from video_community_analysis.csv
    1: 0.8077,
    2: 0.9444
}

# Calc emotional "coherence" = % neutral + top emotion
for comm_id in sorted(nlp_df['Video_Community'].unique()):
    if isinstance(comm_id, (int, float)) and not pd.isna(comm_id):
        comm_data = nlp_df[nlp_df['Video_Community'] == comm_id]
        emotions = comm_data['Predominant_Emotion'].value_counts()
        
        # Coherence = sum of top 3 emotions (more coherent = more dominated by few emotions)
        coherence_pct = 100 * emotions.head(3).sum() / len(comm_data)
        echo_strength = community_cohesion.get(int(comm_id), 0)
        
        print(f"Community {int(comm_id)}:")
        print(f"  Echo Chamber Strength: {echo_strength:.4f}")
        print(f"  Emotional Coherence: {coherence_pct:.1f}%")
        print(f"  Interpretation: {'Strong echo chamber' if echo_strength > 0.90 else 'MODERATE' if echo_strength > 0.80 else 'WEAK'}")
        print()

# Emotion by community matrix
emotion_matrix = pd.crosstab(nlp_df['Video_Community'], nlp_df['Predominant_Emotion'], normalize='index') * 100
emotion_matrix.to_csv('community_emotion_matrix.csv')
print("\nSaved to: community_emotion_matrix.csv (emotion % by community)")

# Topic by community matrix
topic_matrix = pd.crosstab(nlp_df['Video_Community'], nlp_df['Topic_Cluster'], normalize='index') * 100
topic_matrix.to_csv('community_topic_matrix.csv')
print("Saved to: community_topic_matrix.csv (topic % by community)")

# Summary stats
summary_stats = []
for comm_id in sorted(nlp_df['Video_Community'].unique()):
    comm_data = nlp_df[nlp_df['Video_Community'] == comm_id]
    if len(comm_data) > 0:
        summary_stats.append({
            'Community_ID': comm_id,
            'Num_Comments': len(comm_data),
            'Unique_Users': comm_data['User'].nunique(),
            'Num_Topics': comm_data['Topic_Cluster'].nunique(),
            'Dominant_Emotion': comm_data['Predominant_Emotion'].value_counts().index[0],
            'Neutral_Pct': 100 * (comm_data['Predominant_Emotion'] == 'neutral').sum() / len(comm_data),
            'Annoyance_Pct': 100 * (comm_data['Predominant_Emotion'] == 'annoyance').sum() / len(comm_data),
            'Chi2_Pvalue': p_val,
            'Echo_Chamber_Strength': community_cohesion.get(comm_id, 0)
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('community_sentiment_integration_summary.csv', index=False)
print("Saved to: community_sentiment_integration_summary.csv (integration summary)")

# Topic-emotion by community
coupling_data = []
for comm_id in sorted(nlp_df['Video_Community'].unique()):
    comm_data = nlp_df[nlp_df['Video_Community'] == comm_id]
    for topic in comm_data['Topic_Cluster'].unique()[:5]:  # Top 5 topics per community
        topic_data = comm_data[comm_data['Topic_Cluster'] == topic]
        top_emotion = topic_data['Predominant_Emotion'].value_counts().index[0]
        coupling_data.append({
            'Community_ID': comm_id,
            'Topic_ID': int(topic),
            'Comment_Count': len(topic_data),
            'Dominant_Emotion': top_emotion,
            'Neutral_Pct': 100 * (topic_data['Predominant_Emotion'] == 'neutral').sum() / len(topic_data)
        })

coupling_df = pd.DataFrame(coupling_data)
coupling_df.to_csv('community_topic_emotion_coupling.csv', index=False)
print("Saved to: community_topic_emotion_coupling.csv (topic-emotion associations)")
print("\nKey Insight:")
if p_val < 0.05:
    print(f"Communities exhibit significant differences in emotional profiles")
    print(f"(Chi-square p-value: {p_val:.2e})")
else:
    print(f"No statistically significant emotional differences between communities")
