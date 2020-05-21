#!/usr/bin/env python
# coding: utf-8

# In[3]:




# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch #deal with tensors


import torch.nn as nn
import pickle
from collections import Counter

import torchvision.transforms as transforms #transforms the dataset into variouys
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable

from PIL import Image
import torch.optim as optim
import os
import spacy
from torch.utils.data import Dataset


# In[2]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error


# In[23]:


torch.manual_seed(0)

#Cuda algorithms
torch.backends.cudnn.deterministic = True 


# In[24]:


os.chdir('/Users/heoun/Documents/rnn_data/')


# In[25]:


#Good Post for understanding conceptual aspects about LSTMs
#https://colah.github.io/posts/2015-08-Understanding-LSTMs/


# ### Data Cleaning + Importing

# In[26]:


df = pd.read_csv('reviews.csv')
df.head()


# In[27]:


# df['Title'] = df['Title'].fillna('')
# df['Review Text'] = df['Review Text'].fillna('')
df['review'] = df['Title'] + " " + df['Review Text']
#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
df['Rating'] = df['Rating'].apply(lambda x: zero_numbering[x])
df[df['review'] == True]


# In[28]:


df.dropna(inplace = True)


# In[29]:


df1 = df[[ 'review', 'Rating']]
df1.columns = ('review','rating')
df1[df1['review'] == '']


# In[30]:


df1.shape


# In[31]:


df1.dropna(inplace= True)


# In[32]:


df[df['Review Text'] == '']


# In[33]:


df1[df1['review'] == '']


# In[34]:


df1.to_csv('df.csv', index = False)


# ![title](https://miro.medium.com/max/1400/0*pq30znWeMlVXUxxX.png)

# In[35]:


#Even though we’re going to be dealing with text, since our model can only work with numbers, we convert the input into a sequence of numbers where each number represents a particular word (more on this in the next section).


# In[36]:


#embedding layer - our model can only work with #'s so convert the input to #'s where each number represents a word with closeness


#input 3x8
x = torch.tensor([[1,2, 12,34, 56,78, 90,80],
                 [12,45, 99,67, 6,23, 77,82],
                 [3,24, 6,99, 12,56, 21,22]])

#nn.Embedding = a layer that takes the vocab size and desired word-vec length as input
    #in an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space.
model1 = nn.Embedding(100, 7, padding_idx=0)
model2 = nn.LSTM(input_size=7, hidden_size=3, num_layers=1, batch_first=True)


#The LSTM layer outputs three things:
    # The consolidated output — of all hidden states in the sequence
    # Hidden state of the last LSTM unit — the final output
    # Cell state
    
    
out1 = model1(x) #initalize the instance of 100x7 with the 0th index representing the padding
out2 = model2(out1) #the output from model 1 is then plugged into an LSTM of model 2

out, (ht, ct) = model2(out1) #the 3x8 runs through the embedding which then has additional 7 features witth a lstm of hidden size of 3. end result ends in a 3x3
print(ht)
#if the first element in our input’s shape has the batch size - specify it true


# In[37]:


#Try using RMSE - root mean squared error b/c we want to see how close the model is able to predict to the actual value bc of residual comparison.


# In[ ]:


#tokenization
tok = spacy.load('en')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]


# In[ ]:


#count number of occurences of each word
counts = Counter()
for index, row in df1.iterrows():
    counts.update(tokenize(row['review']))


# In[ ]:


#deleting infrequent words
print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))


# In[ ]:


#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)


# In[ ]:


def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


# In[ ]:


df1['encoded'] = df1['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
df1.head()


# In[5]:


X = list(df1['encoded'])
y = list(df1['rating'])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


# In[ ]:


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


# In[ ]:





# In[ ]:


train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)


# In[ ]:


def train_model(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


# In[ ]:


batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)


# In[ ]:


class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


# In[ ]:


model_fixed =  LSTM_fixed_len(vocab_size, 50, 50)


# In[ ]:


train_model(model_fixed, epochs=30, lr=0.01)


# In[ ]:


train_model(model_fixed, epochs=30, lr=0.01)


# In[ ]:


train_model(model_fixed, epochs=30, lr=0.01)


# In[ ]:





# In[ ]:





# In[ ]:




