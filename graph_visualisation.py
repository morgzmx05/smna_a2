import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import warnings
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

nlp_df = pd.read_csv('housing_crisis_nlp_analysed.csv')
video_metrics = pd.read_csv('video_network_metrics.csv')
community_analysis = pd.read_csv('video_community_analysis.csv')
emotion_matrix = pd.read_csv('community_emotion_matrix.csv', index_col=0)
topic_matrix = pd.read_csv('community_topic_matrix.csv', index_col=0)
coupling_df = pd.read_csv('community_topic_emotion_coupling.csv')
sentiment_summary = pd.read_csv('community_sentiment_integration_summary.csv')

# Function to fetch video metadata from YouTube API
def get_video_metadata(youtube, video_ids):
    """Fetches title and channel for a list of video IDs."""
    metadata = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            response = youtube.videos().list(part="snippet", id=",".join(batch)).execute()
            for item in response.get("items", []):
                metadata[item["id"]] = {
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"]
                }
        except Exception as e:
            print(f"Error fetching metadata: {e}")
    return metadata

# Initialize YouTube API
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

communities = {
    0: ['JMin2Czkx1M', 'uUSLmYPHkCc', 'jV6BprD7toY', 'c37eEtRLPjU', 'qHAXbR7SYP8', 'GAu1Y-5EPeM', 'PNRl8TsMa9w', 'LuEaGNXlh1E'],
    1: ['4IihsPhVd7c', 'H-o_F6nKOro', 'HRP1t-P0Ljw', 'a8iNcHeBUCU', 'feffoaDAgO0', 'g29yvU-6ep4', 'vSfec19YOPg', 'h3Ny2J6eNsM', 'glVqpSLyrZc', 'x6e1PGPdepM', 'RJDVeLa7uXg', 'yNH91o9hP8w', 'RfzDHc5dF9Q'],
    2: ['2B4wf5E44bw', 'JqSbECqD7bc', 'u71XD1Bt6Wo', 'PQn41fzVry0', 'w2XCq5VyITI', 'IqKIlLS7ln4', 'kOJ8PoOTFhI', 'vWlh41jJ2Ww', 'hk9CbqKEl-w']
}

video_to_community = {}
for comm_id, videos in communities.items():
    for video in videos:
        video_to_community[video] = comm_id

video_metrics['Community'] = video_metrics['Video_ID'].map(video_to_community)

# Load BERT topic labels with meaningful names derived from example comments
bert_topic_labels = {
    -1: 'Uncategorized',
    0: 'Immigration & Labor',
    1: 'Political Frustration',
    2: 'Immigration Crisis',
    3: 'Govt & Property Conflicts',
    4: 'Direct Replies',
    5: 'Govt Corruption',
    6: 'Anti-Immigration',
    7: 'Labor Party Criticism',
    8: 'Sydney Housing',
    9: 'International Comparisons',
    10: 'Video Feedback',
    11: 'Market Analysis',
    12: 'Albanese Blame',
    13: 'Taxation Issues',
    14: 'Video Timestamps',
    15: 'Socialism Debate',
    16: 'Off-topic Chat',
    17: 'Labor & Greens Blame',
    18: 'Rental Crisis',
    19: 'Video Commentary',
    20: 'Greens Support',
    21: 'Investment & Tax',
    22: 'Monetary Policy',
    23: 'Financial Coaching',
    24: 'Conspiracy (NWO)',
    25: 'Policy Solutions',
    26: 'Wealth Inequality',
    27: 'NZ References',
    28: 'Housing Quality',
    29: 'Emoji Reactions',
    30: 'Real Estate Agents',
    31: 'Video Analysis',
    32: 'Housing Rights Debate',
    33: 'Generational Divide',
    34: 'Supply vs Demand',
    35: 'Construction & Regulation',
    36: 'Land Availability',
    37: 'Chinese Investment',
    38: 'Humor & Sarcasm',
    39: 'Investment Strategy',
    40: 'Tax Policy Discussion',
    41: 'Market Price Movement',
    42: 'Greed as Root Cause',
    43: 'Singapore Model',
    44: 'Indian/Asian Immigration',
    45: 'International Living',
    46: 'Local Area Discussion',
    47: 'Wage & Income',
    48: 'Strata & Apartments',
    49: 'Negative Gearing',
    50: 'Banking Criticism',
    51: 'Homelessness',
    52: 'Employment & Wages',
    53: 'Albanese Immigration',
    54: 'Tenant Rights',
    55: 'Foreign Ownership',
    56: 'Politician Accountability',
    57: 'Homelessness Reality',
    58: 'Urban Living Costs',
    59: 'Market Crash Speculation',
    60: 'Interest Rates',
    61: 'Australian Dream',
    62: 'Supply Solutions',
    63: 'Sydney vs Other Cities',
    64: 'Bank Blame',
    65: 'Debt & Government',
    66: 'Call to Action',
    67: 'Media Criticism',
    68: 'Content Creator Praise',
    69: 'RBA Criticism',
    70: 'Solutions Discussion'
}

# Vid network graph
G_video = nx.Graph()
for idx, row in video_metrics.iterrows():
    G_video.add_node(row['Video_ID'], 
                    community=int(row['Community']),
                    betweenness=row['Betweenness_Centrality'],
                    comments=row['Num_Comments'])

# Add edges (from OG network)
for idx, row in community_analysis.iterrows():
    videos_str = row['Videos']
    if isinstance(videos_str, str):
        videos_list = [v.strip() for v in videos_str.split(';')]
        for i, v1 in enumerate(videos_list):
            for v2 in videos_list[i+1:]:
                if v1 in G_video and v2 in G_video:
                    G_video.add_edge(v1, v2)

pos = nx.spring_layout(G_video, k=2, iterations=50, seed=42)

fig, ax = plt.subplots(figsize=(14, 10))

community_colors = {0: 'salmon', 1: 'lightskyblue', 2: 'turquoise'}
node_colors = [community_colors[G_video.nodes[node]['community']] for node in G_video.nodes()]
node_sizes = [500 + 2000 * G_video.nodes[node]['betweenness'] for node in G_video.nodes()]

nx.draw_networkx_edges(G_video, pos, alpha=0.2, ax=ax)
nx.draw_networkx_nodes(G_video, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)

# Fetch video metadata and create labels using CHANNEL NAMES
metadata = get_video_metadata(youtube, list(G_video.nodes()))

# Create labels using channel names (shorter, more meaningful for network analysis)
labels = {}
for video_id in G_video.nodes():
    if video_id in metadata:
        labels[video_id] = metadata[video_id]['channel']
    else:
        labels[video_id] = video_id[:8]  # Fallback to first 8 chars of ID

nx.draw_networkx_labels(G_video, pos, labels, font_size=7, ax=ax)

# Generate legend labels dynamically from community_analysis
legend_elements = []
for idx, row in community_analysis.iterrows():
    comm_id = row['Community_ID']
    num_videos = row['Num_Videos']
    color = {0: 'salmon', 1: 'lightskyblue', 2: 'turquoise'}.get(comm_id, 'gray')
    legend_elements.append(Patch(facecolor=color, label=f'Community {int(comm_id)} ({int(num_videos)} videos)'))
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
ax.set_title('Video Network Graph: Communities by Shared Audiences\n(Node size = betweenness centrality)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('vis_1_video_network.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_1_video_network.png")
plt.close()

# Emotional distribution by community
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
community_names = {0: 'Community 0\n(Annoyed)', 1: 'Community 1\n(Curious)', 2: 'Community 2\n(Critical)'}

for idx, comm_id in enumerate([0, 1, 2]):
    emotion_data = emotion_matrix.loc[comm_id].nlargest(10)
    colors_grad = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(emotion_data)))
    
    axes[idx].barh(range(len(emotion_data)), emotion_data.values, color=colors_grad)
    axes[idx].set_yticks(range(len(emotion_data)))
    axes[idx].set_yticklabels(emotion_data.index)
    axes[idx].set_xlabel('Percentage (%)', fontsize=11)
    axes[idx].set_title(community_names[comm_id], fontsize=12, fontweight='bold')
    axes[idx].set_xlim(0, 60)
    
    for i, v in enumerate(emotion_data.values):
        axes[idx].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

plt.suptitle('Emotional Profiles by Video Community\n(Top 10 emotions)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('vis_2_emotion_distri.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_2_emotion_distri.png")
plt.close()

# Emotional heatmap
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(emotion_matrix.iloc[:, :15], annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5)
ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
ax.set_ylabel('Community', fontsize=12, fontweight='bold')
ax.set_title('Emotional Heatmap: Communities vs Top 15 Emotions\n(Darker = higher %, lighter = lower %)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('vis_3_emotion_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_3_emotion_heatmap.png")
plt.close()

# Topic distribution by community (using BERT-identified topics)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, comm_id in enumerate([0, 1, 2]):
    topic_data = topic_matrix.loc[comm_id][topic_matrix.loc[comm_id] > 0].nlargest(10)
    colors_grad = plt.cm.viridis(np.linspace(0.2, 0.9, len(topic_data)))
    
    axes[idx].bar(range(len(topic_data)), topic_data.values, color=colors_grad, edgecolor='black', linewidth=1.2)
    axes[idx].set_xticks(range(len(topic_data)))
    # Use BERT-identified topics
    topic_labels = [f'{bert_topic_labels.get(int(t), f"Topic {int(t)}")}' for t in topic_data.index]
    axes[idx].set_xticklabels(topic_labels, fontsize=9, rotation=45, ha='right')
    axes[idx].set_ylabel('Percentage (%)', fontsize=11)
    axes[idx].set_title(f'Community {comm_id}\n({len(communities[comm_id])} videos)', fontsize=12, fontweight='bold')
    
    for i, v in enumerate(topic_data.values):
        axes[idx].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('Topic Distribution by Community (Top 10) - BERT Topics', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('vis_4_topic_distri.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_4_topic_distri.png")
plt.close()

# Centrality comparson
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data
centrality_data = []
for idx, row in video_metrics.iterrows():
    centrality_data.append({
        'Community': int(row['Community']),
        'Degree': row['Degree_Centrality'],
        'Betweenness': row['Betweenness_Centrality']
    })

centrality_df = pd.DataFrame(centrality_data)

# Degree centrality
sns.boxplot(data=centrality_df, x='Community', y='Degree', ax=axes[0], palette=['salmon', 'lightskyblue', 'turquoise'])
axes[0].set_ylabel('Degree Centrality', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Community', fontsize=11, fontweight='bold')
axes[0].set_title('Degree Centrality Distribution', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(['Community 0', 'Community 1', 'Community 2'])

# Betweenness centrality
sns.boxplot(data=centrality_df, x='Community', y='Betweenness', ax=axes[1], palette=['salmon', 'lightskyblue', 'turquoise'])
axes[1].set_ylabel('Betweenness Centrality', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Community', fontsize=11, fontweight='bold')
axes[1].set_title('Betweenness Centrality (Bridge Strength)', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(['Community 0', 'Community 1', 'Community 2'])

plt.suptitle('Network Position Analysis by Community', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('vis_5_centrality_analysis.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_5_centrality_analysis.png")
plt.close()

# Echo chamber vs emotional coherence (dynamically computed from sentiment summary)
fig, ax = plt.subplots(figsize=(14, 6))

# Extract values from sentiment summary
echo_strength = []
emotional_coherence = []
community_names = []
for idx, row in sentiment_summary.iterrows():
    if 'Community' in str(row.get(sentiment_summary.columns[0], '')):
        comm_id = int(row[sentiment_summary.columns[0]].split()[-1])
        community_names.append(f'Community {comm_id}')
        # Find echo chamber strength and emotional coherence values
        for col in sentiment_summary.columns:
            if 'echo' in col.lower() or 'strength' in col.lower():
                echo_strength.append(float(row[col]))
            elif 'coherence' in col.lower():
                emotional_coherence.append(float(row[col]))

# Fallback if dynamic extraction fails
if not echo_strength:
    echo_strength = [0.8929, 0.8077, 0.9444]
    emotional_coherence = [65.6, 64.5, 65.1]
    community_names = ['Community 0', 'Community 1', 'Community 2']

x = np.arange(len(community_names))
width = 0.35

bars1 = ax.bar(x - width/2, echo_strength, width, label='Echo Chamber Strength', color='salmon', alpha=0.8, edgecolor='black', linewidth=1.2)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, emotional_coherence, width, label='Emotional Coherence (%)', color='lightskyblue', alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Echo Chamber Strength', fontsize=12, fontweight='bold', color='salmon')
ax2.set_ylabel('Emotional Coherence (%)', fontsize=12, fontweight='bold', color='lightskyblue')
ax.set_xticks(x)
ax.set_xticklabels(community_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.0)
ax2.set_ylim(60, 70)
ax.set_title('Echo Chamber Strength vs Emotional Coherence\n(Higher coherence = fewer dominant emotions)', 
             fontsize=13, fontweight='bold', pad=15)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}' if height < 2 else f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)  # Manual spacing
plt.savefig('vis_6_echo_chamber_coherence.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
print("Saved to: vis_6_echo_chamber_coherence.png")
plt.close()

# Community size & engagement (computed dynamically from data)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Compute community stats from data
community_stats = []
for comm_id in sorted(set(video_to_community.values())):
    comm_videos = [v for v, c in video_to_community.items() if c == comm_id]
    comm_users = nlp_df[nlp_df['Video_ID'].isin(comm_videos)]['User'].nunique()
    comm_comments = len(nlp_df[nlp_df['Video_ID'].isin(comm_videos)])
    color = {0: 'salmon', 1: 'lightskyblue', 2: 'turquoise'}.get(comm_id, 'gray')
    
    community_stats.append({
        'name': f'Community {comm_id}',
        'videos': len(comm_videos),
        'users': comm_users,
        'comments': comm_comments,
        'color': color
    })

# Create one subplot per community with all metrics grouped
metrics_names = ['Videos', 'Users', 'Comments']
x_pos = np.arange(len(metrics_names))
bar_width = 0.6

for idx, stats in enumerate(community_stats):
    ax = axes[idx]
    values = [stats['videos'], stats['users'], stats['comments']]
    
    # Normalize for visualization (videos are much smaller, so scale for readability)
    display_values = [stats['videos'] * 500, stats['users'], stats['comments']]
    
    bars = ax.bar(x_pos, display_values, bar_width, color=stats['color'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        if i == 0:  # Videos
            label = str(val)
        else:  # Users and Comments (format with commas)
            label = f"{val:,}"
        
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, label, 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold')
    ax.set_title(f'Community {idx}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Community Scale & Engagement Metrics (All metrics per community)', 
             fontsize=15, fontweight='bold', y=0.98)
plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.1, wspace=0.35)
plt.savefig('vis_7_community_scale.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_7_community_scale.png")
plt.close()

# Top creator vids
fig, ax = plt.subplots(figsize=(12, 7))

top_videos = video_metrics.nlargest(10, 'Betweenness_Centrality')
colors_by_community = ['salmon' if int(c) == 0 else 'lightskyblue' if int(c) == 1 else 'turquoise' 
                       for c in top_videos['Community']]

# Fetch video metadata and create labels using CHANNEL NAMES
metadata = get_video_metadata(youtube, top_videos['Video_ID'].tolist())

bars = ax.barh(range(len(top_videos)), top_videos['Betweenness_Centrality'].values, color=colors_by_community, 
               edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(top_videos)))

# Create labels using channel names
channel_labels = []
for video_id in top_videos['Video_ID'].values:
    if video_id in metadata:
        channel_labels.append(metadata[video_id]['channel'])
    else:
        channel_labels.append(video_id[:8])  # Fallback to first 8 chars of ID

ax.set_yticklabels(channel_labels)
ax.set_xlabel('Betweenness Centrality (Bridge Strength)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Bridge Videos (Cross-Community Connectors)\n(Higher centrality = connects more communities)', 
             fontsize=13, fontweight='bold')

for i, (idx, row) in enumerate(top_videos.iterrows()):
    ax.text(row['Betweenness_Centrality'] + 0.002, i, f"{row['Num_Comments']} comments", 
            va='center', fontsize=9)

legend_elements = [
    Patch(facecolor='salmon', label='Community 0'),
    Patch(facecolor='lightskyblue', label='Community 1'),
    Patch(facecolor='turquoise', label='Community 2')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('vis_8_top_creators.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_8_top_creators.png")
plt.close()

# Summary stats table (dynamically computed from all data)
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Build summary data dynamically
summary_data = [['Metric', 'Community 0', 'Community 1', 'Community 2']]

# Add row: # Videos
videos_row = ['# Videos'] + [str(len([v for v, c in video_to_community.items() if c == i])) for i in [0, 1, 2]]
summary_data.append(videos_row)

# Add row: # Users
users_row = ['# Users']
for comm_id in [0, 1, 2]:
    comm_videos = [v for v, c in video_to_community.items() if c == comm_id]
    num_users = nlp_df[nlp_df['Video_ID'].isin(comm_videos)]['User'].nunique()
    users_row.append(f"{num_users:,}")
summary_data.append(users_row)

# Add row: # Comments
comments_row = ['# Comments']
for comm_id in [0, 1, 2]:
    comm_videos = [v for v, c in video_to_community.items() if c == comm_id]
    num_comments = len(nlp_df[nlp_df['Video_ID'].isin(comm_videos)])
    comments_row.append(f"{num_comments:,}")
summary_data.append(comments_row)

# Add row: Avg Comments/Video
avg_comments_row = ['Avg Comments/Video']
for comm_id in [0, 1, 2]:
    comm_videos = [v for v, c in video_to_community.items() if c == comm_id]
    num_comments = len(nlp_df[nlp_df['Video_ID'].isin(comm_videos)])
    avg = int(num_comments / len(comm_videos)) if comm_videos else 0
    avg_comments_row.append(str(avg))
summary_data.append(avg_comments_row)

# Add row: Neutral %
neutral_row = ['Neutral %']
for comm_id in [0, 1, 2]:
    neutral_pct = emotion_matrix.loc[comm_id, 'neutral'] if 'neutral' in emotion_matrix.columns else 0
    neutral_row.append(f"{neutral_pct:.1f}%")
summary_data.append(neutral_row)

# Add row: Top Emotion
top_emotion_row = ['Top Emotion']
for comm_id in [0, 1, 2]:
    top_emotion = emotion_matrix.loc[comm_id].idxmax()
    top_pct = emotion_matrix.loc[comm_id].max()
    top_emotion_row.append(f"{top_emotion.title()} ({top_pct:.1f}%)")
summary_data.append(top_emotion_row)

# Add row: Echo Chamber
echo_row = ['Echo Chamber']
echo_labels = {0: 'Mod', 1: 'Weak', 2: 'Strong'}  # placeholder labels
for comm_id in [0, 1, 2]:
    echo_val = echo_strength[comm_id] if comm_id < len(echo_strength) else 0
    echo_row.append(f"{echo_val:.4f} ({echo_labels.get(comm_id, 'Unknown')})")
summary_data.append(echo_row)

# Add row: Emotional Coherence
coherence_row = ['Emotional Coherence']
for comm_id in [0, 1, 2]:
    coherence_val = emotional_coherence[comm_id] if comm_id < len(emotional_coherence) else 0
    coherence_row.append(f"{coherence_val:.1f}%")
summary_data.append(coherence_row)

# Build color table dynamically
colors_table = []
for i, row in enumerate(summary_data):
    if i == 0:  # Header row
        colors_table.append(['#2C3E50', 'salmon', 'lightskyblue', 'turquoise'])
    else:
        colors_table.append(['#F5F5F5' if i % 2 == 0 else '#FFFFFF'] * 4)

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor(colors_table[0][i])
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(summary_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F5F5F5')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

plt.title('Video Community Analysis Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('vis_9_summary_table.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_9_summary_table.png")
plt.close()
