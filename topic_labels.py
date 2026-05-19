import pandas as pd

#load CSV generated
df = pd.read_csv("housing_crisis_nlp_analysed.csv")

# the top keywords/examples for each topic to help name them
# We group by the Topic_Cluster and grab the first 5 comments from each
topic_summary = df.groupby('Topic_Cluster')['Text'].apply(lambda x: " | ".join(str(i)[:100] + "..." for i in x.head(5))).reset_index()

# Rename columns for clarity
topic_summary.columns = ['Topic_ID', 'Example_Comments_Preview']

# Save as a reference
topic_summary.to_csv("topic_dictionary.csv", index=False)

print("Topic Summary Generated")
print("Open 'topic_dictionary.csv' to see what people are talking about in each cluster.")