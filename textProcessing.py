from datetime import datetime
import re
import emoji
import contractions
from collections import Counter
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.lda_model
from wordcloud import WordCloud
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation


class textProcessing:
        def __init__(self, tokeniser, lStopwords):
            self.tokeniser = tokeniser
            self.lStopwords = lStopwords
        
        def processText(self, text):
            if not isinstance(text, str):
                return []
            regexDigit = re.compile(r"^\d+[\d./,:%-]*\d*$")
            regexHttp = re.compile(r"^http")
            regexMention = re.compile(r"^@")
            text = emoji.replace_emoji(text, replace='')
            text = text.lower()
            text = contractions.fix(text)
            tokens = self.tokeniser.tokenize(text)
            tokensStripped = [tok.strip() for tok in tokens]
            currText = []
            for tok in tokensStripped:
                if tok not in self.lStopwords and regexDigit.match(tok) is None and regexHttp.match(tok) is None and regexMention.match(tok) is None:
                    currText.append(tok)
                    
            return currText
        
        def halfPreprocessText(self, text):
            if not isinstance(text, str):
                return []
            regexDigit = re.compile(r"^\d+[\d./,:%-]*\d*$")
            regexHttp = re.compile(r"^http")
            regexMention = re.compile(r"^@")
            tokens = self.tokeniser.tokenize(text)
            tokensStripped = [tok.strip() for tok in tokens]
            currText = []
            for tok in tokensStripped:
                if regexDigit.match(tok) is None and regexHttp.match(tok) is None and regexMention.match(tok) is None:
                    currText.append(tok)
                    
            return currText
        
        def returnUniqueWords(self, df, columnName):
            all_tokens = [token for tokens in df[columnName] for token in tokens]
            word_freq = Counter(all_tokens)
            return word_freq 
        
        def outputGraphWordFreq(self, counter, numTerms):
            y = [count for tag, count in counter.most_common(numTerms)]
            x = range(1, len(y) + 1)

            #figure size   
            plt.figure(figsize=(30, 10))
            plt.bar(x, y)
            plt.xticks(x, [tag for tag, count in counter.most_common(numTerms)], rotation=90)
            plt.title("Term frequency distribution")
            plt.ylabel('# of words with term frequency')
            plt.xlabel('Term frequency')
            plt.show()
            
        def outputGraphTimeSeries(self, df):
            dateCounts = []
            for x in df['Date']:
                dateCounts.append(datetime.strptime(x[:10], '%Y-%m-%d'))
            dateFreq = Counter(dateCounts)
            x = sorted(dateFreq.keys())
            y = [dateFreq[date] for date in x]
            plt.figure(figsize=(30, 10))
            plt.plot(x, y)
            plt.title("Time Series of Date Frequencies")
            plt.ylabel("Frequency")
            plt.xlabel("Date")
            plt.show()
            
        def outputGraphEmotionPieChart(self, df, top=8):
            emotionFreq = Counter(df['Predominant_Emotion'])
            
            topEmotions = dict(emotionFreq.most_common(top))
            otherCount = sum(v for k, v in emotionFreq.items() if k not in topEmotions)
            if otherCount > 0:
                topEmotions['other'] = otherCount
            
            labels = list(topEmotions.keys())
            sizes = list(topEmotions.values())
            
            plt.figure(figsize=(10, 10))
            plt.pie(
                sizes, 
                labels=labels, 
                autopct='%1.1f%%', 
                startangle=140,
                pctdistance=0.85,   # push percentages outward
                labeldistance=1.1,  # push labels outward
            )
            plt.title("Distribution of Emotions")
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        
        
        
        
        def display_topics(self, model, featureNames, numTopWords):
    

    
            for topicId, lTopicDist in enumerate(model.components_):
                print('Topic %d:' % (topicId + 1))
                print(' '.join([featureNames[i] for i in lTopicDist.argsort()[:-numTopWords - 1:-1]]))
                
        def displayWordcloud(self, model, featureNames):   
            normalisedComponents = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

            
            plotRowNum = 1
            plotColNum = 3
            plt.figure(figsize=(20, 6))
            for topic, topicProb in enumerate(normalisedComponents):
                lWordProb = {featureNames[i]: topicProb[i] for i in range(len(featureNames))}
                wc = WordCloud(background_color='white', max_words=20, width=800, height=400)
                wc.generate_from_frequencies(lWordProb)
                
                plt.subplot(plotRowNum, plotColNum, topic + 1)
                plt.imshow(wc, interpolation='bilinear')
                plt.title(f'Topic {topic + 1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('wordcloud.png', dpi=150)
            plt.show()
    
        def getLDAModel(self, data, topicNum, wordNumToDisplay, featureNum):
            comments = []
            for _, x in data.iterrows():
                comments.append(' '.join(x['processedTokens']))


            tfVectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=featureNum, stop_words='english')
            tf = tfVectorizer.fit_transform(comments)


            tfFeatureNames = tfVectorizer.get_feature_names_out()

            ldaModel = LatentDirichletAllocation(n_components=topicNum, max_iter=10, learning_method='online').fit(tf)


            self.display_topics(ldaModel, tfFeatureNames, wordNumToDisplay)

            panel = pyLDAvis.lda_model.prepare(ldaModel, tf, tfVectorizer, mds='mmds')
            pyLDAvis.save_html(panel, 'ldaVis.html')

            self.displayWordcloud(ldaModel, tfFeatureNames)
                
