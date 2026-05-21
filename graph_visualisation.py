import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import warnings
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

topic_names = {
    -1: 'Uncategorized',
    0: 'Immigration & Demand',
    1: 'Anti-Immigration',
    2: 'Casual/Off-topic',
    3: 'Real Estate Analysis',
    4: 'Geographic Issues',
    5: 'Government Policy',
    6: 'Dismissive Tone',
    7: 'Labor Blame',
    8: 'Anti-Government',
    9: 'Political Voting',
    10: 'International Compare',
    11: 'Rental Crisis',
    12: 'Video-Specific',
    13: 'Systemic Failure',
    14: 'Ideology Debate',
    15: 'Tax Criticism',
    16: 'Supply/Demand Partisan',
    17: 'Dismissive/Sarcastic',
    18: 'Greens Support',
    19: 'National Pessimism',
    20: 'Wealth Inequality',
    21: 'Party Attribution',
    22: 'Conspiracy/NWO',
    23: 'Monetary Policy',
    24: 'Finance Coaching',
    25: 'Video Quality',
    26: 'Avocado Toast Joke',
    27: 'Construction Regulation',
    28: 'Immigration Policy',
    29: 'Supply vs Demand',
    30: 'Banking System Criticism',
    31: 'Real Estate Agent Blame',
    32: 'Land/Geography Arguments',
    33: 'Miscellaneous',
    34: 'Cross-country Chat'
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
nx.draw_networkx_labels(G_video, pos, font_size=8, ax=ax)

legend_elements = [
    Patch(facecolor='salmon', label='Community 0 (8 videos)'),
    Patch(facecolor='lightskyblue', label='Community 1 (13 videos)'),
    Patch(facecolor='turquoise', label='Community 2 (9 videos)')
]
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

# Topic distribution by community
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, comm_id in enumerate([0, 1, 2]):
    topic_data = topic_matrix.loc[comm_id][topic_matrix.loc[comm_id] > 0].nlargest(10)
    colors_grad = plt.cm.viridis(np.linspace(0.2, 0.9, len(topic_data)))
    
    axes[idx].bar(range(len(topic_data)), topic_data.values, color=colors_grad, edgecolor='black', linewidth=1.2)
    axes[idx].set_xticks(range(len(topic_data)))
    # Use actual topic names instead of T1, T2, etc.
    topic_labels = [f'{topic_names.get(int(t), f"T{int(t)}")}' for t in topic_data.index]
    axes[idx].set_xticklabels(topic_labels, fontsize=9, rotation=45, ha='right')
    axes[idx].set_ylabel('Percentage (%)', fontsize=11)
    axes[idx].set_title(f'Community {comm_id}\n({len(communities[comm_id])} videos)', fontsize=12, fontweight='bold')
    
    for i, v in enumerate(topic_data.values):
        axes[idx].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('Topic Distribution by Community (Top 10)', 
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

# Echo chamber vs emotional coherence (Note: Graph is funky, zoom in -> scroll down, ss for report)
fig, ax = plt.subplots(figsize=(14, 6))

community_names = ['Community 0', 'Community 1', 'Community 2']
echo_strength = [0.8929, 0.8077, 0.9444]
emotional_coherence = [65.6, 64.5, 65.1]

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

# Community size & engagement
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

community_stats = [
    {'name': 'Community 0', 'videos': 8, 'users': 4391, 'comments': 6040, 'color': 'salmon'},
    {'name': 'Community 1', 'videos': 13, 'users': 4237, 'comments': 7209, 'color': 'lightskyblue'},
    {'name': 'Community 2', 'videos': 9, 'users': 5679, 'comments': 7814, 'color': 'turquoise'}
]

for idx, stats in enumerate(community_stats):
    # Vids
    axes[0].bar(idx, stats['videos'], color=stats['color'], edgecolor='black', linewidth=1.2)
    axes[0].text(idx, stats['videos'] + 0.2, str(stats['videos']), ha='center', va='bottom', fontweight='bold')
    
    # Users
    axes[1].bar(idx, stats['users'], color=stats['color'], edgecolor='black', linewidth=1.2)
    axes[1].text(idx, stats['users'] + 100, f"{stats['users']:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Comments
    axes[2].bar(idx, stats['comments'], color=stats['color'], edgecolor='black', linewidth=1.2)
    axes[2].text(idx, stats['comments'] + 150, f"{stats['comments']:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)

axes[0].set_ylabel('Number of Videos', fontsize=11, fontweight='bold')
axes[0].set_title('Videos per Community', fontsize=12, fontweight='bold')
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['Comm 0', 'Comm 1', 'Comm 2'])

axes[1].set_ylabel('Unique Users', fontsize=11, fontweight='bold')
axes[1].set_title('Users per Community', fontsize=12, fontweight='bold')
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['Comm 0', 'Comm 1', 'Comm 2'])

axes[2].set_ylabel('Total Comments', fontsize=11, fontweight='bold')
axes[2].set_title('Comments per Community', fontsize=12, fontweight='bold')
axes[2].set_xticks([0, 1, 2])
axes[2].set_xticklabels(['Comm 0', 'Comm 1', 'Comm 2'])

plt.suptitle('Community Scale & Engagement Metrics', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('vis_7_community_scale.png', dpi=300, bbox_inches='tight')
print("Saved to: vis_7_community_scale.png")
plt.close()

# Top creator vids
fig, ax = plt.subplots(figsize=(12, 7))

top_videos = video_metrics.nlargest(10, 'Betweenness_Centrality')
colors_by_community = ['salmon' if int(c) == 0 else 'lightskyblue' if int(c) == 1 else 'turquoise' 
                       for c in top_videos['Community']]

bars = ax.barh(range(len(top_videos)), top_videos['Betweenness_Centrality'].values, color=colors_by_community, 
               edgecolor='black', linewidth=1.2)
ax.set_yticks(range(len(top_videos)))
ax.set_yticklabels(top_videos['Video_ID'].values)
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

# Summary stats table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

summary_data = [
    ['Metric', 'Community 0', 'Community 1', 'Community 2'],
    ['# Videos', '8', '13', '9'],
    ['# Users', '4,391', '4,237', '5,679'],
    ['# Comments', '6,040', '7,209', '7,814'],
    ['Avg Comments/Video', '755', '555', '869'],
    ['Neutral %', '52.3%', '52.3%', '53.3%'],
    ['Top Emotion', 'Annoyance (7.4%)', 'Curiosity (6.7%)', 'Disappointment (5.0%)'],
    ['Echo Chamber', '0.8929 (Mod)', '0.8077 (Weak)', '0.9444 (Strong)'],
    ['Emotional Coherence', '65.6%', '64.5%', '65.1%']
]

colors_table = [['#E8E8E8']*4] + [['#FFFFFF']*4 for _ in range(len(summary_data)-1)]
colors_table[0] = ['#2C3E50', 'salmon', 'lightskyblue', 'turquoise']

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
