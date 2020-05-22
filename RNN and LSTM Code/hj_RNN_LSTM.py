#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch #deal with tensors
import torch.nn as nn #optim , Dataset , and DataLoader to help you create and train neural networks

from collections import Counter #dict class for counting hashable objects

import torchvision #basically has all the important datasets and mdoels and transfomration operations for CV
import torchvision.transforms as transforms #transforms the dataset into various ways
import torchvision.datasets as datasets #batching and working with data is already done for us so it has prepared data
from torch.autograd import Variable #variable wraps a trnesor.provides backwards method to perform backprop 
#For example, to backpropagate a loss function to train model parameter x, we use a variable loss to store the value computed by a loss function

import torch.optim as optim #optimizations that are used

import spacy #industrial level word of splitting between verb and noun
from torch.utils.data import Dataset, DataLoader #torchvision is actually a subset of the datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence #pack a tensor containing padded sequence of variable length. 

import string #way to process stnadard strings
import torch.nn.functional as F
import re #gives PerL like functionality, finding 
from sklearn.metrics import mean_squared_error


# In[95]:


#set seed
torch.manual_seed(0)
#Cuda algorithms
torch.backends.cudnn.deterministic = True 


# In[96]:


os.chdir('/Users/heoun/Documents/rnn_data/')


# In[97]:


#Good Post for understanding conceptual aspects about LSTMs
#https://colah.github.io/posts/2015-08-Understanding-LSTMs/


# ### Data Cleaning + Importing

# In[98]:


df = pd.read_csv('reviews.csv')
df.head(2)


# In[99]:


df['review'] = df['Title'] + " " + df['Review Text']
#changing ratings to 0-numbering
zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
df['Rating'] = df['Rating'].apply(lambda x: zero_numbering[x])
#df[df['review'] == ''] #this wont work because NAN values are nothing
df[df['review'].isnull()]


# There are indeed null values and will need to be cleaned.

# In[100]:


df.shape


# In[101]:


df.dropna(inplace = True)
df.shape


# In[102]:


df['length'] = df['review'].apply(lambda x: len(x.split()))
np.mean(df['length'])


# In[103]:


df1 = df[['review', 'Rating']]
df1.columns = ('review','rating')
df1[df1['review'].isnull()]


# In[104]:


df1.shape

df1.to_csv('df.csv', index = False)
# Three gates in a LSTM Cell - Check OneNote

# ![title](https://miro.medium.com/max/1400/0*pq30znWeMlVXUxxX.png)

# In[105]:


#Even though we’re going to be dealing with text, since our model can only work with numbers, we convert the input into a sequence of numbers where each number represents a particular word (more on this in the next section).


# ### High Level LSTM Example
# 

# In[132]:



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


# In[133]:


#Try using RMSE - root mean squared error b/c we want to see how close the model is able to predict to the actual value bc of residual comparison.


# ### Preprocessing - convert words to index/vectors

# In[134]:


#tokenization - industrial level word of splitting between verb and noun

#https://realpython.com/natural-language-processing-spacy-python/

#It allows you to identify the basic units in your text. These basic units are called tokens. Tokenization is useful because it breaks a text into meaningful units. These units are used for further analysis, like part of speech tagging.
tok = spacy.load('en') #loading the english


def spacy_tokenize (text): #removing punctuation, special characters, and lower casing the text before running the tokenizer 
    text = re.sub(r"[^\x00-\x7F]+", " ", text) #subsitute and remove any of the following at the left
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower()) #lowercase without punctuation
    return [token.text for token in tok.tokenizer(nopunct)] #splits the text into sentences - depends on the ask #https://www.youtube.com/watch?v=Y90BJzUcqlI 


# In[135]:


#count number of occurences of each word to see the count
counts = Counter()
for index, row in df1.iterrows():
    counts.update(spacy_tokenize(row['review'])) #spacy_tokenize(row['review']) groups the like words and spacy_tokenizes them indivivduually the updates command keeps track ofthe amount of counts

#deleting infrequent words less than 2
print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))
# In[136]:


#creating vocabulary of words that can be referenced - this indexes the words appropriately
vocab2index = {"":0, "t":1} #creating that initial dictionary
words = ["", "t"] #words that are going to be indexed
for i in counts: #counts now has a list of words with the counts for each. words is now iterating through the count
    vocab2index[i] = len(words) #takes the word length and stores it
    words.append(i) #also appends the word into words. so now we have a length of words in the vocab 2 index and the word itself


# In[137]:


df1['review'][2]


# In[138]:


tokenized1 = spacy_tokenize(df1['review'][2]) #any word is tokenized from the text inputted
print(tokenized1[1], tokenized1[2])
encoded = np.zeros(70, dtype=int)
enc1 = np.array([vocab2index.get(x, vocab2index["t"]) for x in tokenized1])
print(enc1[1], enc1[2], enc1[3])
min(70, len(enc1))
encoded[:70] = enc1[:70]
encoded


# In[141]:


def encode_sentence(text, vocab2index, N=70): #set to 70 since a sentence is approx 70 max
    tokenized = spacy_tokenize(text) #any word is tokenized from the text inputted
    encoded = np.zeros(N, dtype=int) #creates a np list of the mean 
    
    enc1 = np.array([vocab2index.get(x, vocab2index["t"]) for x in tokenized]) #access a value from the dictionary. #this pulls the values that were tokenized from each row and turns them into numerical values
    
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length] 
    return encoded, length


# In[145]:


df1['encoded'] = df1['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index ))) #creating an array out of the encoding sentence 
df1['encoded']


# In[146]:


X = list(df1['encoded'])
y = list(df1['rating'])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


# In[147]:


# Data Processor Loader
# torch.utils.data Dataset
class ReviewsDataset(Dataset): #dataloader - dataset is straighforward since encodings are already in the input df. Also printed the output of the length of input sequence since LSTMs have to depend on whether to use a path or not.
    def __init__(self, X, Y):
        self.X = X #initalize x and y
        self.y = Y
        
    def __len__(self): #aka you are using the xtrain and tells you how its doing in length
        return len(self.y) #this is the len of the sentence
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]#get the x as a 32 int and the y back. this gives you the proper feed of current, 1-5 and how it sdoing on the scale.
        #https://pytorch.org/docs/master/generated/torch.from_numpy.html


# In[156]:


train_df = ReviewsDataset(X_train, y_train)
test_df = ReviewsDataset(X_valid, y_valid)


# ### Optimizer & Parameters - SGD

# In[ ]:


batch_size = 5000
vocab_size = len(words)

optimizer = torch.optim.SGD(parameters, lr= 0.001)
parameters = filter(lambda p: p.requires_grad, model.parameters()) #arams (iterable) – an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized.


#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

#an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is
#Batching the data
# Shuffling the data
# Load the data in parallel using multiprocessing workers.
train_dl = DataLoader(train_df, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(test_df, batch_size=batch_size)


# In[157]:


vocab_size


# In[ ]:


class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) #https://www.quora.com/What-is-the-embedding-layer-in-LSTM-long-short-term-memory
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) #LSTM to properly retain information and move forward
        
        self.linear = nn.Linear(hidden_dim, 5) #lniear layer
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


# In[ ]:


def train_model(model, epochs=4): #creating the actual training of the model
    
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            
            #Optimizer and backprop
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))


# In[149]:


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


# In[151]:





# In[152]:


model_fixed =  LSTM_fixed_len(vocab_size, 50, 50)


# In[153]:


train_model(model_fixed, epochs=30, lr=0.01)


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:





# In[ ]:




