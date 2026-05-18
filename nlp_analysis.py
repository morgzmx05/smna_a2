import pandas as pd
from bertopic import BERTopic
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

#load csv file
df = pd.read_csv("housing_crisis_nlp_data.csv")


# BERTopic (Advanced Topic Modeling) 
# We use a custom vectorizer to remove common noise words like 'video' or 'just' 
vectorizer_model = CountVectorizer(stop_words="english")

df['Text'] = df['Text'].fillna('').astype(str).str.strip()
df = df[df['Text'] != ''].reset_index(drop=True)

print("Running BERTopic")
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=False,
    verbose=True
)

#run the modelonly on the 'Text' column

topics, _ = topic_model.fit_transform(df['Text'].astype(str))
df['Topic_Cluster'] = topics

# retrieve labels for e.g Topic 1: immigration, rent, prices
topic_info = topic_model.get_topic_info()
print(topic_info.head(10))

# RoBert a (Deep Emotion Classification)
# This looks for 28 specific emotions
print("Running Emotion Analysis")
emotion_pipe = pipeline(
    "text-classification", 
    model="SamLowe/roberta-base-go_emotions", 
    top_k=1
)

def get_emotion_label(text):
    try:
        # BERT has a 512 character limit, hence truncate to avoid errors
        result = emotion_pipe(str(text)[:512])
        return result[0][0]['label']
    except:
        return "neutral"

df['Predominant_Emotion'] = df['Text'].apply(get_emotion_label)

# Save the final analysed file
df.to_csv("housing_crisis_nlp_analysed.csv", index=False)
print("-Task Complete- File saved as housing_crisis_nlp_analysed.csv")