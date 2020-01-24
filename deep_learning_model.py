#!/usr/bin/env python
# coding: utf-8

# In[31]:


import json
import numpy as np
import re
import os
import nltk
import pandas as pd
from keras.utils import np_utils


from keras.layers import Dense ,LSTM,concatenate,Input,Flatten
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Activation, merge, Flatten, Reshape
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers


# # Problem Forumlation

# ### Model is provided two inputs question_vector and document_vector and asked to predict the start and end index of answer in the document

# # Load word embeddings

# In[32]:


embedding_path = "./GloVe/glove.840B.300d.txt"
embedding_path = "/Users/rajatjain/Documents/new_webapp/react-app-kubernetes/backend/src/app/data/glove/glove.6B.100d.txt"


# In[33]:


train = pd.read_json('./data/train-v1.1.json')


# In[63]:


documents = []
questions = []
answers = []
titles = []
answer_start_indexs = []
answer_end_indexs = []
def get_attributes(item):
    data = item['data']
    title = data['title']
    for paragraph in data['paragraphs']:
        for qas in paragraph['qas']:
            answer = qas['answers'][0]['text']
            answer_start_index = qas['answers'][0]['answer_start']
            answer_end_index = answer_start_index + len(answer.split(' ')) - 1
            answers.append(qas['answers'][0]['text'])
            questions.append(qas['question'])
            documents.append(paragraph['context'])
            answer_start_indexs.append(answer_start_index)
            answer_end_indexs.append(answer_end_index)
            
            titles.append(title)
            
def build_dataframe(train):
    train.apply(get_attributes, axis = 1)
    train_df = pd.DataFrame({
    'documents':documents,
    'questions': questions,
    'answers': answers,
    'titles': titles,
     'answer_end_indexs': answer_end_indexs,
    'answer_start_indexs': answer_start_indexs
})
    return train_df
    
train_df = build_dataframe(train)
train_df = train_df.head(5000)


# In[64]:


train_df.head()


# In[43]:


def get_max_length(sentences):
    max_length = 0
    for sentence in sentences:
        length_of_sentence = len(sentence)
        if length_of_sentence > max_length:
            max_length = length_of_sentence
    return max_length


# # Extract Entities

# In[67]:


train_df = train_df.head(2000)
documents = list(train_df["documents"])
questions = list(train_df["questions"])
answer_start_indexs = train_df["answer_start_indexs"].values
answer_end_indexs = train_df["answer_end_indexs"].values
sentences = documents + questions


# In[68]:


questions = train_df['questions'].values
answers = train_df['answers'].values
documents = train_df['documents'].values


# # Vectorize Question, Answer and Context

# In[70]:


vectorized_data = []
def vectorize(item):
    tokenizer = Tokenizer(
    num_words = 20000,
    filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~'   
)
        
    documents = list(item["documents"])
    questions = list(item["questions"])
    answers = list(item['answers'])
    start_index = list(item['answer_start_indexs'])
    end_index = list(item['answer_end_indexs'])
    sentences = documents + questions
    
    tokenizer.fit_on_texts(sentences)
    questions_tokenized = tokenizer.texts_to_sequences(questions)
    answers_tokenized = tokenizer.texts_to_sequences(answers)
    documents_tokenized = tokenizer.texts_to_sequences(documents)
    
    questions_padded = pad_sequences(questions_tokenized, maxlen = 80, padding = 'post')
    answers_padded = pad_sequences(answers_tokenized, maxlen = 1405, padding = 'post')
    documents_padded = pad_sequences(documents_tokenized, maxlen = 1405, padding = 'post')
    for i in range(0, len(documents)):
        vectorized_data.append([questions_padded[i], answers_padded[i], documents_padded[i], start_index[i], end_index[i] ])
    
train_df.groupby('documents').apply(vectorize)
vectorized_data = pd.DataFrame(vectorized_data)
vectorized_data.rename(columns = {0: 'question_vector', 1: 'answers_vector', 2: 'documents_vector', 3: 'answer_start_indexs', 4: 'answer_end_indexs' },inplace = True)


# In[71]:


vectorized_data.head()


# # Model Architecture

# In[72]:


question_input = Input(shape=(80,), dtype='int32', name='question_input')
context_input =  Input(shape=(1405,), dtype='int32', name='context_input')

questionEmbd = Embedding(output_dim=100, input_dim=20000,
                         mask_zero=False, 
                         input_length=80, trainable=False)(question_input)


contextEmb = Embedding(output_dim=100, input_dim=20000,
                         mask_zero=False,
                         input_length=1405, trainable=False)(context_input)


# In[73]:


Q = Bidirectional(LSTM(80, return_sequences=True))(questionEmbd)
D = Bidirectional(LSTM(40, return_sequences=True))(contextEmb)
Q_flatten = Flatten()(Q)
D_flatten = Flatten()(D)
merged = concatenate([D_flatten, Q_flatten])


# In[74]:


output1 = Dense(1,activation='sigmoid')(merged)
l2_merged = concatenate([merged, output1])
output2 = Dense(1,activation='sigmoid')(l2_merged)

model = Model(inputs=[question_input,context_input], output = [output1,output2])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()


# # Model Fit

# In[77]:


questions_padded = np.array(vectorized_data['question_vector'].values.tolist())
documents_padded = np.array(vectorized_data['documents_vector'].values.tolist())
answer_begin = np.array(vectorized_data['answers_vector'].values.tolist())
answer_start_indexs = np.array(vectorized_data['answer_start_indexs'].values.tolist())
answer_end_indexs = np.array(vectorized_data['answer_end_indexs'].values.tolist())


# In[61]:


history = model.fit([questions_padded, documents_padded],[answer_start_indexs,  answer_end_indexs] ,
                    epochs=10,
                    batch_size=300)

