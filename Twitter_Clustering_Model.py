# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 20:39:16 2017

@author: akshp
"""

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
from nltk import bigrams

# load nltk's English stopwords as variable called 'stopwords'
#stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
    
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', '\n']


#grammar = r"""
#    NBAR:
#        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
#        
#    NP:
#        {<NBAR>}
#        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
#"""

grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}        
"""

#grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
#grammar = """
#         NP: {<DT|PP\$>?<JJ>*<NN>}
#        {<NNP>+}
#        {<NN>+}
#        """
good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    #tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
#    tokens = [word.lower() for word in preprocess(text) 
#              if word not in stop and
#              not word.startswith(('#', '@')) 
#              ]

    tokens = [word.lower() for sent in nltk.sent_tokenize(text) 
              for word in preprocess(sent) 
              if word not in stop and
              not word.startswith(('#', '@'))
              and not word.startswith("'") 
              and not word.endswith("'")
              and "'" not in word]
    
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) 
              for word in preprocess(sent) 
              if word not in stop and
              not word.startswith(('#', '@')) 
              and not word.startswith("'") 
              and not word.endswith("'")
              and "'" not in word]
    
#    tokens = [word.lower() for word in preprocess(text) 
#              if word not in stop and
#              not word.startswith(('#', '@'))]
    
    filtered_tokens = []
    
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):    
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem_word(word)
    #word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stop)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term
        
def get_chunks(text):
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',r'',text)
    text = text.lower()
    toks = tokenize_only(text)
    postoks = nltk.tag.pos_tag(toks)
    chunker = nltk.RegexpParser(grammar)
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    allwords_tokenized = []
    for term in terms:
        phrase = []
        for word in term:
            phrase.append(word)
        key = ' '.join(phrase)
        if key in ecdict:
            if ecdict.get(key) != '':
                allwords_tokenized.append(ecdict.get(key))
        #else:
            #allwords_tokenized.append(key)
        #allwords_tokenized.append(str(phrase))
    return allwords_tokenized


def get_words(text):
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',r'',text)
    text = text.lower()
    toks = tokenize_only(text)
    tagged_words = nltk.tag.pos_tag(toks)    
    
    # filter on certain POS tags and lowercase all words
    allwords_tokenized = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop
                  ]
    return allwords_tokenized
        

def get_phrases_and_terms(text,candidates='chunks'):        
    boc_texts = []
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',r'',text)
    text = text.lower()
    if candidates == 'chunks':
        boc_texts.extend(get_chunks(text))
    elif candidates == 'words':
        boc_texts.extend(get_words(text))
    return boc_texts



import csv
dirname = 'C:/Users/akshp/Google Drive/Predict 453 Text Analytics/Project Twitter Text Analytics/Data/'
 
import pandas as pd
corpus = []
df = pd.read_csv(dirname+'TweetsTrumpPresidency.csv')
Tweets = df[(df.RetweetCount > 0)].Text #you can also use df['column_name']
for tweet in Tweets:
    corpus.append(tweet)
#print(Tweets)

dirname = 'C:/Users/akshp/Google Drive/Predict 453 Text Analytics/Project Twitter Text Analytics/'
ecdict = {}
with open(dirname+'terms.csv', mode='r') as infile:
    reader = csv.reader(infile)
    ecdict = {row[1]:row[2] for row in reader}

totalvocab_tokenized=[]
for text in corpus:
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',r'',text)
    text = text.lower()
    totalvocab_tokenized.extend(get_chunks(text))
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_tokenized)  
    


from sklearn.feature_extraction.text import TfidfVectorizer

#tfidf_vectorizer = TfidfVectorizer( stop_words=stop,tokenizer=tokenize, ngram_range=(1,1))
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=100,
                                   min_df=2, stop_words=stop,
                                 use_idf=True, tokenizer=get_phrases_and_terms, ngram_range=(1,1))


%time tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
dense = tfidf_matrix.todense()

from sklearn.metrics.pairwise import cosine_similarity
#cosine similarity if document 1 with others
cosine_similarity(tfidf_matrix[1], tfidf_matrix)
dist = 1 - cosine_similarity(tfidf_matrix)

#Using LSA to check for clustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
svd = TruncatedSVD(n_components=2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

%time tfidf_matrix_lsa = lsa.fit_transform(tfidf_matrix)
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


#hierarchical document clustering
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

from matplotlib import pyplot as plt
plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right");

               
#k-means with td-idf matrix
from sklearn.cluster import KMeans
num_clusters = 2
km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1,verbose=2)
%time tfidf_Xfm = km.fit_transform(tfidf_matrix)
cluster_labels = km.fit_predict(tfidf_matrix)
clusters = km.labels_.tolist()

#display clusters

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#create data frame that has the result of the LSA plus the cluster numbers
df = pd.DataFrame(dict(x=tfidf_matrix_lsa[:,0], y=tfidf_matrix_lsa[:,1], label=clusters))
#group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,              
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], size=8)     
    
plt.show() #show the plot 


#another way of showing clusters

# 2nd Plot showing the actual clusters formed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) #
#colors = cm.spectral(clusters.astype(float) / num_clusters)
ax.scatter(tfidf_matrix_lsa[:,0], tfidf_matrix_lsa[:,1],s=30, lw=0, alpha=0.7)
# Labeling the clusters
centers = km.cluster_centers_
# Draw white circles at cluster centers
ax.scatter(centers[:, 0], centers[:, 1],
            marker='o', c="white", alpha=1, s=200)

for i, c in enumerate(centers):
    ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

ax.set_title("The visualization of the clustered data.")
ax.set_xlabel("Feature space for the 1st feature")
ax.set_ylabel("Feature space for the 2nd feature")

plt.show()

tweetdata = { 'tweets': corpus, 'cluster':clusters}
frame =  pd.DataFrame(tweetdata, index = [clusters] , columns = ['tweets','cluster'])

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :25]:
        print(' %s' % terms[ind], end='')
    print()

#for i in range(num_clusters):
#    print("Cluster %d words:" % i, end='')
#    
#    for ind in order_centroids[i, :20]: #replace 6 with n words per cluster
#        
#        print(' %s' % vocab_frame.ix[ind], end=',')
#    print() #add whitespace
#    print() #add whitespace
#    
#    print("Cluster %d titles:" % i, end='')
#    for title in frame.ix[i]['title']:
#        print(' %s,' % title, end='')
#    print() #add whitespace
#    print() #add whitespace
    
from sklearn.metrics import silhouette_samples, silhouette_score
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
print("For n_clusters =", num_clusters,
      "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(tfidf_matrix, cluster_labels)

fig, ax1 = plt.subplots(figsize=(17, 9))
y_lower = 10
for i in range(num_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.spectral(float(i) / num_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# 2nd Plot showing the actual clusters formed
fig, ax2 = plt.subplots(figsize=(17, 9))
colors = cm.spectral(cluster_labels.astype(float) / num_clusters)
ax2.scatter(tfidf_Xfm[:, 0], tfidf_Xfm[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors)

# Labeling the clusters
centers = km.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1],
            marker='o', c="white", alpha=1, s=200)

for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")
plt.show()


#LSA applied to tfidf-matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
svd = TruncatedSVD(n_components=2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

%time tfidf_matrix_lsa = lsa.fit_transform(tfidf_matrix)
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

#from sklearn.cluster import KMeans
#num_clusters = 2
#km_lsa = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1,verbose=2)
#%time km.fit(tfidf_matrix_lsa)
#clusters_lsa = km.labels_.tolist()
#original_space_centroids = svd.inverse_transform(km_lsa.cluster_centers_)
#order_centroids = original_space_centroids.argsort()[:, ::-1]


#Topic modelling with LDA
import gensim, nltk
def lda_score_keyphrases_by_tfidf(texts, candidates='chunks'):    
    boc_texts = []
    for text in texts:
        text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',r'',text)
        text = text.lower()
        if candidates == 'chunks':
            boc_texts.append(get_chunks(text))
        elif candidates == 'words':
            boc_texts.append(get_words(text))
    #make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    dictionary.filter_extremes(no_below=0.4, no_above=0.8)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf, dictionary, corpus


#LDA model
tfidif, dictionary, bow = lda_score_keyphrases_by_tfidf(corpus,'chunks')
#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)

print(dictionary.token2id)
print(bow[0])
ldamodel = gensim.models.ldamodel.LdaModel(bow, num_topics=2, id2word = dictionary, passes=100, update_every=5,chunksize=100)

ldamodel.show_topics()

import numpy as np
topics_matrix = ldamodel.show_topics(formatted=False, num_words=20)
topics_matrix