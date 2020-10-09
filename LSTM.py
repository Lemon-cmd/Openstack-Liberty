#!/usr/bin/env python
# coding: utf-8

# In[77]:


import keras
from keras.models import Sequential
import numpy as np
import nltk


# In[78]:


corpus = nltk.corpus.treebank.tagged_sents()
corp_size = len(corpus)
taggedW_size = len(nltk.corpus.treebank.tagged_words())


# In[79]:


#separate tags and words into 2 lists
sentences, tsentences = [], []

for tsent in corpus:
    words, tags = zip(*tsent)
    sentences.append(words)
    tsentences.append(tags)


# In[80]:


#split data into train and test for cross validation
from sklearn.model_selection import train_test_split

train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tsentences, test_size=0.25) 


# In[81]:


#enumerate train_sentencnes, test_sentences and tags
words, tags = set([]), set([])

for s in train_sentences:
    for w in s:
        words.add(w.lower())

for ts in tsentences:
    for t in ts:
        tags.add(t)


# In[82]:


word_id = {w: i + 2 for i,w in enumerate(list(words))}
word_id['-PAD-'] = 0 #value for padding
word_id['-OOV-'] = 1 #out of vocabulary

tag_id = {t: i + 1 for i, t in enumerate(list(tags))}
tag_id['-PAD-'] = 0 
id_tag = {tag_id[key]:key for key in tag_id}
tags_size = len(tag_id)


# In[83]:


train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word_id[w.lower()])
        except KeyError:
            s_int.append(word_id['-OOV-'])
 
    train_sentences_X.append(s_int)
    
for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word_id[w.lower()])
        except KeyError:
            s_int.append(word_id['-OOV-'])
 
    test_sentences_X.append(s_int)
    
for s in train_tags:
    train_tags_y.append([tag_id[t] for t in s])
    
for s in test_tags:
    test_tags_y.append([tag_id[t] for t in s])


# In[84]:


train_tags_y[0]


# In[85]:


#Max Length of the train sequence and perhaps the test sequence
MAX_LENGTH_TRAIN = len(max(train_sentences_X, key=len))
MAX_LENGTH_TEST = len(max(test_sentences_X, key=len))
MAX_LENGTH = max(MAX_LENGTH_TRAIN, MAX_LENGTH_TEST)


# In[86]:


#pad the inputs as NN can only deal with fixed inputs :/ sadly
from keras.preprocessing.sequence import pad_sequences
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post') 


# In[87]:


train_tags_y[0]


# In[88]:


from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word_id), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag_id))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()


# ### Derive Y from sequences

# In[89]:


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


# In[90]:


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences
 


# In[91]:


cat_train_tags_y = to_categorical(train_tags_y, tags_size)
cat_train_tags_y[0].shape


# In[93]:


model.fit(train_sentences_X, cat_train_tags_y, shuffle=True, batch_size=128, epochs=20, validation_split=0.30, use_multiprocessing=True)


# In[94]:

cat_test_tags_y = to_categorical(test_tags_y, tags_size)

# In[95]:


loss, accuracy = model.evaluate(test_sentences_X, cat_test_tags_y)
print("Evaulation => Loss: {0} Accuracy: {1} ".format(loss, accuracy))

