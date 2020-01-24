#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import pickle
from textblob import TextBlob
import torch

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')


# # Dowload Nltk Data

# In[2]:


import nltk
nltk.download('punkt')


# # Load Infersent Pre-trained Model

# In[7]:


from InferSent.models import InferSent
V = 1
MODEL_PATH = './encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = './Glove/glove.840B.300d.txt'
infersent.set_w2v_path(W2V_PATH)


# # Read Data

# In[8]:


train = pd.read_json('./data/train-v1.1.json')


# In[9]:


contexts = []
questions = []
answers = []
titles = []

def get_attributes(item):
    data = item['data']
    title = data['title']
    for paragraph in data['paragraphs']:
        for qas in paragraph['qas']:
            answers.append(qas['answers'][0]['text'])
            questions.append(qas['question'])
            contexts.append(paragraph['context'])
            titles.append(title)
            
def build_dataframe(train):
    train.apply(get_attributes, axis = 1)
    train_df = pd.DataFrame({
    'contexts':contexts,
    'questions': questions,
    'answers': answers,
    'titles': titles
})
    return train_df
    
train_df = build_dataframe(train)
train_df = train_df.head(5000)


# In[10]:


train_df.shape


# # Split Paragraph into Sentences

# In[11]:


train_df['sentences'] = train_df['contexts'].apply(lambda x : [item.raw for item in TextBlob(x).sentences ])


# # Get Target

# In[12]:


def get_target(item):
    """ Builds the target using the index number of answer in the list of sentences
    """
    for index, sentence in enumerate( item['sentences']):
        if item['answers'] in sentence:
            return index
    return 0
    
train_df['target'] = train_df.apply(get_target, axis = 1)


# In[13]:


def get_all_sentences(sentences):
    all_sentences = []
    sentences = sentences.tolist()
    for context_sentences in sentences:
        for setence in context_sentences:
            all_sentences.append(setence)
        
    all_sentences = list(dict.fromkeys(all_sentences))
    return all_sentences


# # Generate Vocab

# In[14]:


paras = list(train_df["contexts"].drop_duplicates().reset_index(drop= True))
blob = TextBlob(" ".join(paras))
sentences = get_all_sentences(train_df['sentences'])
infersent.build_vocab(sentences, tokenize=True)


# # Build Embeddings

# In[15]:


# Sentence Embeddings
dict_embeddings = {}
for i in range(len(sentences)):
    dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)[0]
  
# Question Embeddings
questions = list(train_df["questions"])    
for i in range(len(questions)):
    dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)[0]
    


# # Save Embeddings

# In[156]:


# Todo
# This will help to save the computation time


# In[157]:


def get_context_embeddings(item):
    embeddings = []
    for sentence in item.sentences:
        embeddings.append(dict_embeddings[sentence])
    return embeddings


# In[158]:


train_df['question_embedding'] = train_df['questions'].apply(lambda x : dict_embeddings[x])
train_df['context_embedding'] = train_df.apply(get_context_embeddings, axis = 1)


# In[159]:


train_df.head()


# # Generate Features

# In[160]:


from sklearn.metrics.pairwise import euclidean_distances


# In[161]:


def get_metric(item, metric):
    result = []
    for i in range(0,len(item.sentences)):
        question_embedding = [item.question_embedding]
        sentence_embedding = [item['context_embedding'][i]]

        if metric == 'cosine_similarity':
            metric = cosine_similarity(question_embedding, sentence_embedding)
            
        if metric == 'euclidean':
            metric = euclidean_distances(question_embedding, sentence_embedding)  

        result.append(metric[0][0])  
    return result
    


# In[162]:


train_df['cosine_similarity'] = train_df.apply(lambda item : get_metric(item, 'cosine_similarity'), axis = 1)
train_df['euclidean'] = train_df.apply(lambda item : get_metric(item, 'euclidean'), axis = 1)


# In[163]:


train_df.head()


# In[164]:


train_df_copy = train_df.copy()


# # Pad the euclidean and cosine_similarity features

# In[165]:


def find_max_number_of_sentences():
    """
        finds the maximum number of sentences possible in any context
    """
    max_number_of_sentences = 0
    for i in range(0, train_df.shape[0]):
        length = len(train_df.iloc[i].sentences)
        if length > max_number_of_sentences:
            max_number_of_sentences = length  
    return max_number_of_sentences     
    
max_number_of_sentences = find_max_number_of_sentences()


# In[175]:


max_number_of_sentences


# In[166]:


def pad(data, max_length):
    mean = sum(data)/len(data)
    length_of_data = len(data)
    pad_number = max_length - length_of_data
    data = data + [mean]*pad_number
    return data
    


# In[167]:


resultant_data = []
def combine_features(item):
    """
    Pads the euclidean and cosine values for particualr instance and generates resultant dataframe
    for modelling , it has eculidean distance between question and all sentnces and cosine similarity
    between between question and all sentences as well and last feature is the index of the answer in the sentnces
    """
    length_of_sentence = len(item.sentences)
    cosine_similarity = item.cosine_similarity
    euclidean = item.euclidean
    
    if length_of_sentence < max_number_of_sentences:
        euclidean = pad(euclidean, max_number_of_sentences)
        cosine_similarity = pad(cosine_similarity, max_number_of_sentences)
        
    features = euclidean + cosine_similarity + [item.target]    
    resultant_data.append(features)
train_df_copy.apply(combine_features, axis = 1)

resultant_data = pd.DataFrame(resultant_data)


# In[174]:


resultant_data.head()


# In[169]:


X = resultant_data.iloc[:,:-1]
y = resultant_data.iloc[:,-1]


# In[170]:


train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state = 5)


# In[171]:


mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))


# In[172]:


rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
rf.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, rf.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, rf.predict(test_x)))


# In[173]:


MODEL_FILE = "../model.pkl"

mdl = rf.fit(X, y)
joblib.dump(mdl, MODEL_FILE)
from sklearn.externals import joblib

