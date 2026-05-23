import pandas as pd
import networkx as nx
from community import community_louvain
from itertools import combinations
import numpy as np
from collections import defaultdict

# Load nlp data
nlp_df = pd.read_csv("housing_crisis_nlp_data.csv")

print(f"Total comments: {len(nlp_df)}")
print(f"Unique users: {len(nlp_df['User'].unique())}")
print(f"Unique videos: {len(nlp_df['Video_ID'].unique())}")
print(f"Videos: {nlp_df['Video_ID'].unique()[:5]}...")


# Bipartite network
B = nx.Graph()
B.add_nodes_from(nlp_df['User'].unique(), bipartite=0)
B.add_nodes_from(nlp_df['Video_ID'].unique(), bipartite=1)

for _, row in nlp_df.iterrows():
    B.add_edge(row['User'], row['Video_ID'])

print(f"Bipartite network: {B.number_of_nodes()} nodes (users + videos)")
print(f"  - Users: {len(nlp_df['User'].unique())}")
print(f"  - Videos: {len(nlp_df['Video_ID'].unique())}")
print(f"  - Edges: {B.number_of_edges()}")

# Vid projection (vids connected if they share commenters)
videos = set(nlp_df['Video_ID'].unique())
users_by_video = nlp_df.groupby('Video_ID')['User'].apply(set).to_dict()

G_video = nx.Graph()
G_video.add_nodes_from(videos)

# Mutual users for each vid pair (min 3)
video_list = list(videos)
min_mutual_threshold = 3

for video_a, video_b in combinations(video_list, 2):
    users_a = users_by_video[video_a]
    users_b = users_by_video[video_b]
    mutual_users = len(users_a & users_b)
    
    # Decent mutual audience = edge
    if mutual_users >= min_mutual_threshold:
        G_video.add_edge(video_a, video_b, weight=mutual_users)

print(f"Video network created:")
print(f"  - Nodes (videos): {G_video.number_of_nodes()}")
print(f"  - Edges (shared audiences): {G_video.number_of_edges()}")
print(f"  - Threshold: {min_mutual_threshold} mutual users minimum")

# Remove any isolated vids
isolated_videos = list(nx.isolates(G_video))
print(f"  - Isolated videos (no shared audience): {len(isolated_videos)}")
G_video.remove_nodes_from(isolated_videos)

print(f"After removing isolated videos:")
print(f"  - Nodes: {G_video.number_of_nodes()}")
print(f"  - Edges: {G_video.number_of_edges()}")

if G_video.number_of_nodes() == 0:
    print("ERROR: No videos with shared audience!")
    exit()

# Degree centrality (connected vids)
degree_centrality = nx.degree_centrality(G_video)

# Betweenness centrality (vids that bridge comms)
betweenness_centrality = nx.betweenness_centrality(G_video, weight='weight')

# Weighted degree (connections strength)
weighted_degree = dict(G_video.degree(weight='weight'))

# Community detection (Louvain)
communities = community_louvain.best_partition(G_video, weight='weight', randomize=False)

community_sizes = {}
for video, comm_id in communities.items():
    if comm_id not in community_sizes:
        community_sizes[comm_id] = []
    community_sizes[comm_id].append(video)

print(f"\nDetected {len(community_sizes)} video communities:")
for comm_id in sorted(community_sizes.keys()):
    print(f"  Community {comm_id}: {len(community_sizes[comm_id])} videos")

# Modularity
modularity = community_louvain.modularity(communities, G_video, weight='weight')
print(f"\nNetwork Modularity: {modularity:.4f}")

# Vid cluster analysis
video_cluster_analysis = []
for comm_id in sorted(community_sizes.keys()):
    videos_in_community = community_sizes[comm_id]
    
    community_users = set()
    total_comments = 0
    for video in videos_in_community:
        community_users.update(users_by_video[video])
        total_comments += len(nlp_df[nlp_df['Video_ID'] == video])
    
    # Internal connectivity
    subgraph = G_video.subgraph(videos_in_community)
    internal_edges = subgraph.number_of_edges()
    possible_edges = len(videos_in_community) * (len(videos_in_community) - 1) / 2
    internal_density = internal_edges / possible_edges if possible_edges > 0 else 0
    
    avg_mutual_users = np.mean([G_video[u][v]['weight'] for u, v in subgraph.edges()]) if internal_edges > 0 else 0
    
    video_cluster_analysis.append({
        'Community_ID': comm_id,
        'Num_Videos': len(videos_in_community),
        'Unique_Users': len(community_users),
        'Total_Comments': total_comments,
        'Avg_Comments_Per_Video': total_comments / len(videos_in_community),
        'Internal_Density': internal_density,
        'Avg_Mutual_Users': avg_mutual_users,
        'Videos': '; '.join(videos_in_community)
    })

cluster_df = pd.DataFrame(video_cluster_analysis)

print("\nTop 10 Largest Video Communities:")
top_clusters = cluster_df.nlargest(10, 'Unique_Users')[['Community_ID', 'Num_Videos', 'Unique_Users', 'Total_Comments', 'Internal_Density']]
print(top_clusters.to_string())

# Centrality
video_metrics = []
for video in G_video.nodes():
    video_metrics.append({
        'Video_ID': video,
        'Community_ID': communities.get(video, -1),
        'Degree_Centrality': degree_centrality.get(video, 0),
        'Betweenness_Centrality': betweenness_centrality.get(video, 0),
        'Weighted_Degree': weighted_degree.get(video, 0),
        'Num_Commenters': len(users_by_video[video]),
        'Num_Comments': len(nlp_df[nlp_df['Video_ID'] == video])
    })

video_metrics_df = pd.DataFrame(video_metrics)

print("\nMost central videos (highest betweenness/bridges between topics):")
top_videos = video_metrics_df.nlargest(10, 'Betweenness_Centrality')[['Video_ID', 'Betweenness_Centrality', 'Community_ID', 'Num_Commenters', 'Num_Comments']]
print(top_videos.to_string())

titles = pd.read_csv("housing_crisis_video_metadata.csv")[['Video_ID', 'Channel']]
for video in G_video.nodes():
    G_video.nodes[video]['label'] = titles.set_index('Video_ID')['Channel'].to_dict().get(video, video)
    G_video.nodes[video]['community'] = communities.get(video, -1)
    G_video.nodes[video]['degree_centrality'] = degree_centrality.get(video, 0)
    G_video.nodes[video]['betweenness_centrality'] = betweenness_centrality.get(video, 0)
    G_video.nodes[video]['weighted_degree'] = weighted_degree.get(video, 0)
    G_video.nodes[video]['num_commenters'] = len(users_by_video[video])

# Write graphml
nx.readwrite.write_graphml(G_video, 'video_network.graphml', infer_numeric_types=True)
print("Saved to: video_network.graphml")
# Vid metrics saved
video_metrics_df.to_csv("video_network_metrics.csv", index=False)
print("Saved to: video_network_metrics.csv")

# Cluster analysis saved
cluster_df.to_csv("video_community_analysis.csv", index=False)
print("Saved to: video_community_analysis.csv")

# Network stats saved
network_stats = {
    'Metric': [
        'Total Videos',
        'Total Video Connections',
        'Videos with Shared Audience',
        'Isolated Videos (removed)',
        'Network Density',
        'Average Weighted Degree',
        'Number of Video Communities',
        'Network Modularity',
        'Network Connectivity',
        'Min Mutual Users Threshold'
    ],
    'Value': [
        len(videos),
        G_video.number_of_edges(),
        G_video.number_of_nodes(),
        len(isolated_videos),
        nx.density(G_video),
        np.mean(list(weighted_degree.values())),
        len(community_sizes),
        modularity,
        'Connected' if nx.is_connected(G_video) else 'Disconnected',
        min_mutual_threshold
    ]
}

network_stats_df = pd.DataFrame(network_stats)
network_stats_df.to_csv("video_network_statistics.csv", index=False)
print("Saved to: video_network_statistics.csv")


print("\nKey Insights:")
print(f"  - {len(videos)} total videos analyzed")
print(f"  - {G_video.number_of_nodes()} videos with interconnected audiences")
print(f"  - {len(community_sizes)} thematic clusters detected")
print(f"  - Modularity {modularity:.4f} indicates strong topic polarization")
print(f"  - {len(isolated_videos)} videos had no overlapping audience")
print("\nGenerated files:")
print("  1. video_network_metrics.csv - Per-video centrality and community assignment")
print("  2. video_community_analysis.csv - Community-level video cluster analysis")
print("  3. video_network_statistics.csv - Overall network metrics")
