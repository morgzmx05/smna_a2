import pandas as pd
import networkx as nx
from community import community_louvain
import numpy as np

# Load edge list
edges_df = pd.read_csv("housing_crisis_network_data.csv") 

print(f"Total edges: {len(edges_df)}")
print(f"Unique source users: {len(edges_df['Source_User'].unique())}")
print(f"Unique target users: {len(edges_df['Target_User'].unique())}")

# Build directed weighted graph
G_user = nx.DiGraph()
for _, row in edges_df.iterrows():
    G_user.add_edge(row['Source_User'], row['Target_User'], weight=row['Weight'])

print(f"\nUser reply network:")
print(f"  - Nodes (users): {G_user.number_of_nodes()}")
print(f"  - Edges (reply interactions): {G_user.number_of_edges()}")
print(f"  - Density: {nx.density(G_user):.6f}")


# Centrality measures
in_degree = dict(G_user.in_degree(weight='weight'))
out_degree = dict(G_user.out_degree(weight='weight'))
degree_centrality = nx.degree_centrality(G_user)

# Betweenness on undirected version (more meaningful for community bridging)
G_undirected = G_user.to_undirected()
betweenness = nx.betweenness_centrality(G_undirected, weight='weight')

# Community detection requires undirected
communities = community_louvain.best_partition(G_undirected, weight='weight', randomize=False)
modularity = community_louvain.modularity(communities, G_undirected, weight='weight')

community_sizes = {}
for user, comm_id in communities.items():
    community_sizes.setdefault(comm_id, []).append(user)

print(f"\nDetected {len(community_sizes)} user communities")
print(f"Modularity: {modularity:.4f}")
for comm_id in sorted(community_sizes.keys()):
    print(f"  Community {comm_id}: {len(community_sizes[comm_id])} users")

# Per-user metrics
user_metrics = []
for user in G_user.nodes():
    user_metrics.append({
        'User': user,
        'Community_ID': communities.get(user, -1),
        'In_Degree_Weighted': in_degree.get(user, 0),
        'Out_Degree_Weighted': out_degree.get(user, 0),
        'Degree_Centrality': degree_centrality.get(user, 0),
        'Betweenness_Centrality': betweenness.get(user, 0),
    })

metrics_df = pd.DataFrame(user_metrics)
print(metrics_df.describe().loc[['min', '25%', '50%', '75%', 'max']])

print("\nTop 10 most replied-to users (in-degree):")
print(metrics_df.nlargest(10, 'In_Degree_Weighted')[
    ['User', 'In_Degree_Weighted', 'Out_Degree_Weighted', 'Community_ID']
].to_string())

print("\nTop 10 bridge users (betweenness centrality):")
print(metrics_df.nlargest(10, 'Betweenness_Centrality')[
    ['User', 'Betweenness_Centrality', 'Community_ID']
].to_string())

# Attach community + metrics to graph for Gephi export
for user in G_user.nodes():
    G_user.nodes[user]['community'] = communities.get(user, -1)
    G_user.nodes[user]['in_degree_weighted'] = in_degree.get(user, 0)
    G_user.nodes[user]['betweenness_centrality'] = betweenness.get(user, 0)

nx.write_graphml(G_user, 'user_reply_network.graphml', infer_numeric_types=True)
metrics_df.to_csv("user_network_metrics.csv", index=False)

print("\nSaved: user_reply_network.graphml, user_network_metrics.csv")