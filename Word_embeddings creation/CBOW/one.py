#import libaries 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy.spatial.distance import cosine
import pandas as pd
import pickle
from collections import Counter
import os
import json
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re



#load the data from reviews_Movies_and_TV.json file using for loop to iterate through 40,000 sentences in a loop and append it to a list
data = []
with open('reviews_Movies_and_TV.json') as f:
    count = 0 
    for line in f:
        count = count + 1
        data.append(json.loads(line))
        if count == 40000:
            break
# print(len(data))
review_text = []
for i in range(len(data)):
    review_text.append(data[i]['reviewText'])

#tokenize the review text using into sentences
sentences = []
for i in range(len(review_text)):
    sentences.append(word_tokenize(review_text[i]))

#remove the stop words and punctuations from the sentences
stop_words = set(stopwords.words('english'))
for i in range(len(sentences)):
    sentences[i] = [re.sub(r'[^\w\s]', '', word) for word in sentences[i] if word not in stop_words and word != '']
    # sentences[i] = [word for word in sentences[i] if word not in stop_words]

with open('sentences.json', 'w') as f:
    json.dump(sentences, f)

