
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals


# In[2]:


import sklearn
sklearn.__version__


# In[3]:


import re
import nltk
import string
from nltk.stem import WordNetLemmatizer

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                                if contraction_mapping.get(match)                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    
    
from pattern.en import tag
from nltk.corpus import wordnet as wn

# Annotate text tokens with POS tags
def pos_tag_text(text):
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text
    
# lemmatize text based on POS tags    
def lemmatize_text(text):
    
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text
    

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
    
    
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

    

def normalize_corpus(corpus, tokenize=False):
    
    normalized_corpus = []    
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
            
    return normalized_corpus

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):        
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels


# In[4]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


# In[5]:


dataset = pd.read_csv("./testProject_About.labeled.csv", header=None, encoding="utf-8")
dataset['Labels'] = np.where(dataset[1] == 'About', 1, 0)
dataset['Text'] =   dataset[0].astype(unicode)

corpus = dataset['Text'].tolist()
labels = dataset['Labels'].tolist()


# In[6]:


import matplotlib.pyplot as plt
dataset['Labels'].hist()
plt.show()


# In[7]:


train_corpus, test_corpus, train_labels, test_labels = train_test_split(corpus, labels, 
                                                        test_size=0.3, random_state=43)


# In[8]:


print(len(test_corpus), len(test_labels))
print(len(train_corpus),len(train_labels))


# In[9]:


norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus)  


# In[10]:


print(len(norm_test_corpus), len(test_labels))
print(len(norm_train_corpus),len(train_labels))


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer

def bow_extractor(corpus, ngram_range=(1,1)):
    
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
    
    
from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_transformer(bow_matrix):
    
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix
    
    
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_extractor(corpus, ngram_range=(1,1)):
    
    vectorizer = TfidfVectorizer(min_df=1, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
    

import numpy as np    
    
def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector
    
   
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)
    
    
def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
    
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)] 
                   if tfidf_vocabulary.get(word) 
                   else 0 for word in words]    
    word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary: 
            word_vector = model[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)
        
    return feature_vector
    
def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors, 
                                   tfidf_vocabulary, model, num_features):
                                       
    docs_tfidfs = [(doc, doc_tfidf) 
                   for doc, doc_tfidf 
                   in zip(corpus, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                   model, num_features)
                    for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features) 


# In[12]:



import nltk
import gensim


# In[13]:


# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)  
bow_test_features = bow_vectorizer.transform(norm_test_corpus) 

# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)    


# tokenize documents
tokenized_train = [nltk.word_tokenize(text)
                   for text in norm_train_corpus]
tokenized_test = [nltk.word_tokenize(text)
                   for text in norm_test_corpus]  

# build word2vec model                   
model = gensim.models.Word2Vec(tokenized_train,
                               size=500,
                               window=100,
                               min_count=30,
                               sample=1e-3)                  
                   
# averaged word vector features
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,
                                                 model=model,
                                                 num_features=500)                   
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
                                                model=model,
                                                num_features=500)                                                 
                   


# tfidf weighted averaged word vector features
vocab = tfidf_vectorizer.vocabulary_
tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train, 
                                                                  tfidf_vectors=tfidf_train_features, 
                                                                  tfidf_vocabulary=vocab, 
                                                                  model=model, 
                                                                  num_features=500)
tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test, 
                                                                 tfidf_vectors=tfidf_test_features, 
                                                                 tfidf_vocabulary=vocab, 
                                                                 model=model, 
                                                                 num_features=500)


# In[14]:


print(bow_test_features.shape, len(test_labels))
print(bow_train_features.shape,len(train_labels))


# In[138]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, ratio = 'minority')

bow_train_features_bal, bow_train_labels_bal = sm.fit_sample(bow_train_features, train_labels)


# In[139]:


plt.hist(bow_train_labels_bal)


# In[140]:


tfidf_train_features_bal, tfidf_train_labels_bal = sm.fit_sample(tfidf_train_features, train_labels)
avg_wv_train_features_bal, avg_wv_train_labels_bal = sm.fit_sample(avg_wv_train_features, train_labels)
tfidf_wv_train_features_bal, tfidf_wv_train_labels_bal = sm.fit_sample(tfidf_wv_train_features, train_labels)


# In[141]:


from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score

def get_metrics(true_labels, predicted_labels):
    
    print( 'Accuracy', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        2))
    print( 'Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print( 'Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    
def get_log_loss(true_labels, predicted_labels):
    
    print( 'Log Loss', np.round(
                        metrics.log_loss(true_labels, 
                                               predicted_labels),
                        2))
                        

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels 
                                ):
    # build model    
    classifier.fit(train_features, train_labels)
    
    #print("\nLoss on Train set")
    #get_log_loss(true_labels=train_labels, 
    #            predicted_labels=classifier.predict(train_features))
    
    print('\nMetrics on train set')
    get_metrics(true_labels=train_labels, 
                predicted_labels=classifier.predict(train_features))
    
    scores = cross_val_score(classifier, train_features, train_labels,
                             scoring="f1", cv=10)
    
    print("\nF1 Score CrossVal Set")
    display_scores(scores)
    
    #print("\nLoss on CrossVal set")
    #get_log_loss(true_labels=train_labels, 
    #            predicted_labels=classifier.predict(train_features))
    
    
    
    return classifier

def test_predict_evaluate_model(classifier, test_features, test_labels):
    # predict using model
    predictions = classifier.predict(test_features) 
    
    #print("\nLoss on Test set")
    #get_log_loss(true_labels=test_labels, 
    #            predicted_labels=predictions)
    
    # evaluate model prediction performance   
    print('\nMetrics on test set')
    get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)
    
    return predictions
    


# In[142]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

mnb = MultinomialNB()
svm = LinearSVC(C=1, loss="hinge")
logistic = SGDClassifier(loss='log', max_iter=100)



# In[143]:


from sklearn.ensemble import RandomForestClassifier
rnd = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)


# In[144]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier( n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)


# In[145]:


# Multinomial Naive Bayes with bag of words features
mnb_bow_classifier = train_predict_evaluate_model(classifier=mnb,
                                           train_features=bow_train_features_bal,
                                           train_labels=bow_train_labels_bal
                                           )

mnb_bow_predictions = test_predict_evaluate_model(mnb_bow_classifier, test_features=bow_test_features, test_labels=test_labels)


# In[146]:


# Logistic Regression with bag of words features
log_bow_classifier = train_predict_evaluate_model(classifier=logistic,
                                           train_features=bow_train_features_bal,
                                           train_labels=bow_train_labels_bal)

log_bow_predictions = test_predict_evaluate_model(log_bow_classifier, test_features=bow_test_features, test_labels=test_labels)


# In[147]:


# SVC with bag of words features
svm_bow_classifier = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features_bal,
                                           train_labels=bow_train_labels_bal)

svm_bow_predictions = test_predict_evaluate_model(svm_bow_classifier, test_features=bow_test_features, test_labels=test_labels)


# In[152]:


# Multinomial Naive Bayes with tfidf features                                           
mnb_tfidf_classifier = train_predict_evaluate_model(classifier=mnb,
                                           train_features=tfidf_train_features_bal,
                                           train_labels=tfidf_train_labels_bal
                                          )

mnb_tfidf_predictions = test_predict_evaluate_model(mnb_tfidf_classifier, test_features=tfidf_test_features, test_labels=test_labels)


# In[153]:


# Logistic Regression with tfidf features
log_tfidf_classifier = train_predict_evaluate_model(classifier=logistic,
                                           train_features=tfidf_train_features_bal,
                                           train_labels=tfidf_train_labels_bal
                                           )
log_tfidf_predictions = test_predict_evaluate_model(log_tfidf_classifier, test_features=tfidf_test_features, test_labels=test_labels)


# In[154]:


# Support Vector Machine with tfidf features
svm_tfidf_classifier = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_train_features_bal,
                                           train_labels=tfidf_train_labels_bal)

svm_tfidf_predictions = test_predict_evaluate_model(svm_tfidf_classifier, test_features=tfidf_test_features, test_labels=test_labels)


# In[155]:


# RandomForest with tfidf features
rnd_tfidf_classifier = train_predict_evaluate_model(classifier=rnd,
                                           train_features=tfidf_train_features_bal,
                                           train_labels=tfidf_train_labels_bal)

rnd_tfidf_predictions = test_predict_evaluate_model(rnd_tfidf_classifier, test_features=tfidf_test_features, test_labels=test_labels)


# In[156]:


# Boosting with tfidf features
ada_tfidf_classifier = train_predict_evaluate_model(classifier=ada,
                                           train_features=tfidf_train_features_bal,
                                           train_labels=tfidf_train_labels_bal)

ada_tfidf_predictions = test_predict_evaluate_model(ada_tfidf_classifier, test_features=tfidf_test_features, test_labels=test_labels)


# In[157]:


#Logistic Regression with averaged word vector features
log_avgwv_classifier = train_predict_evaluate_model(classifier=logistic,
                                           train_features=avg_wv_train_features_bal,
                                           train_labels=avg_wv_train_labels_bal)

log_avgwv_predictions = test_predict_evaluate_model(log_avgwv_classifier, test_features=avg_wv_test_features, test_labels=test_labels)


# In[159]:


# Support Vector Machine with averaged word vector features
svm_avgwv_classifier = train_predict_evaluate_model(classifier=svm,
                                           train_features=avg_wv_train_features_bal,
                                           train_labels=avg_wv_train_labels_bal
                                           )
svm_avgwv_predictions = test_predict_evaluate_model(svm_avgwv_classifier, test_features=avg_wv_test_features, test_labels=test_labels)


# In[160]:


# Logistic Regression with tfidf weighted averaged word vector features
log_tfidfwv_classifier = train_predict_evaluate_model(classifier=logistic,
                                           train_features=tfidf_wv_train_features_bal,
                                           train_labels=tfidf_wv_train_labels_bal)

log_tfidfwv_predictions = test_predict_evaluate_model(log_tfidfwv_classifier, test_features=tfidf_wv_test_features, test_labels=test_labels)


# In[161]:


# Support Vector Machine with tfidf weighted averaged word vector features
svm_tfidfwv_classifier = train_predict_evaluate_model(classifier=svm,
                                           train_features=tfidf_wv_train_features_bal,
                                           train_labels=tfidf_wv_train_labels_bal
                                           )
svm_tfidfwv_predictions = test_predict_evaluate_model(svm_tfidfwv_classifier, test_features=tfidf_wv_test_features, test_labels=test_labels)


# #### Logistic Regression with tfidf transformation model give best bias-variance tradeoff and is chosen as candidate final model. The final model is chosen by hypertuning this model

# In[163]:


import pandas as pd
cm = metrics.confusion_matrix(test_labels, log_tfidf_predictions)
pd.DataFrame(cm, index=range(0,2), columns=range(0,2))


# In[166]:


logistic = SGDClassifier(loss='log', max_iter=100)

#Grid Search on ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'l1_ratio': [0,1,0.2,0.4,0.6,0.8],
        'alpha': [0.0001, 0.001, 0.01, 0.1],             
    }

sgd_log = SGDClassifier(loss='log', max_iter=100)
rnd_search = RandomizedSearchCV(sgd_log, param_distribs, cv=10, scoring='f1')
rnd_search.fit(tfidf_train_features_bal, tfidf_train_labels_bal)

print('Best Params' + str(rnd_search.best_params_))
print('Best Estimator' + str(rnd_search.best_estimator_))

final_model = rnd_search.best_estimator_
print('Best F1 Score ' + str(np.sqrt(rnd_search.best_score_)))

print('Metrics on Test Set')
get_metrics(true_labels=test_labels, 
                predicted_labels=final_model.predict(tfidf_test_features))

