#!/usr/bin/env python
# coding: utf-8

# ## Import necessary libraries

# In[1]:


import random
import string # to process standard python strings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pip install nltk


# In[3]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only


# ## Reading in the corpus
# 
# For our example,we will be using the Wikipedia page for chatbots as our corpus. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.

# In[4]:


f=open('cc.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase


# ## Tokenisation

# In[5]:


sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
print(sent_tokens[:2])
word_tokens = nltk.word_tokenize(raw)# converts to list of words
print(word_tokens[:8])


# ## Preprocessing
# 
# We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.

# In[6]:


lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# ## Keyword matching
# 
# Next, we shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a greeting response.ELIZA uses a simple keyword matching for greetings. We will utilize the same concept here.

# In[7]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi","It's a pleasure to have you here today!", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# To generate a response from our bot for input questions, the concept of document similarity will be used. We define a function response which searches the user’s utterance for one or more known keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”

# In[8]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you, try rephrasing your question"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# Finally, we will feed the lines that we want our bot to say while starting and ending a conversation depending upon user’s input.

# In[9]:


flag=True
print("CT: My name is Cloud Tech, you can call me CT. I am here to increase your knowledge on cloud computing ^.^\n If you want to exit, type Bye!\n")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("CT: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("CT: "+greeting(user_response))
            else:
                print("CT: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("CT: Bye! take care..")


# In[ ]:




