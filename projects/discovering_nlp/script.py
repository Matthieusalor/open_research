# -*- coding: utf-8 -*-
# %% [markdown]
#
# ## Figuring out Natural Language Processing
# As I have never worked on NLP before, the purpose of this notebook was to playing arround with a dataset and trying to figure out a bunch of stuff on the subject.
# Here we will be working on the IMDB dataset which provides 50k movies text reviews and their corresponding sentiment  "Positive" or "Negative".
#
# Our job will be to find a way to learn some features that can predict the sentiment based on a textual review. 

# %% [markdown]
# ### Load the data
# We will be getting the data from my github repositery. I have downloaded those data from Kaggle https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews .

# %%
import pandas as pd
import requests
from io import StringIO

orig_url='https://drive.google.com/file/d/1Tl9AMNkExM5mFw3xDuIeZ1RiDIEu4Oci/view?usp=sharing'
file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
df = pd.read_csv(csv_raw)
df.head()
# %%
df.iloc[0,0]

# %% [markdown]
# ## Cleaning
# Now that we have the data, and displayed some of those data, we know that there is cleaning to be made. 
#
# For this analysis, I will assume that numbers are meaningless and that we need only words to predict the review. 
# Therefore, we will get rid of : 
# * numbers,
# * html tags,
# * uppercases,
# * any special characters

# %%
# Remove numbers
df['clean_review'] = df['review'].str.replace('\d+', '')
# Remove any <> and everything inside
df['clean_review'] = df['clean_review'].str.replace('<[^<]+?>', '')
# Remove anything that is not alphanumeric
df['clean_review'] = df['clean_review'].str.replace(r'[^A-Za-z0-9 ]+', '')
# Remove any uppercase character
df['clean_review'] = df['clean_review'].str.lower()
# Remove any one character words
df['clean_review'] = df['clean_review'].str.replace(r'\b\w\b', '')
# Remove multiple spaces 
df['clean_review'] = df['clean_review'].str.replace(r'\s+', ' ')
# Strip data
df['clean_review'] = df['clean_review'].str.strip()

df['clean_review'][0]

# %% [markdown]
# ## What direction ? 
# Now, we have a text that seems to be way more clean. 
#
# Obviously, now we will have to create some features out of all these words in order to extract the sentiment. 
# What I mean by that is that we need to create a standardized framework in which any review could fit. The problem with those textual input is that they are of random sizes, and any model that we might create will need inputs of pre-defined sizes.
#
# What we will be using here is some kind of one-hot-encoding technic. The concept is simple, you take a categorical variable and transform it in vector space. ie: 
#
# | category |
# |---|
# | A |
# | B |
# | C | 
#
# | A | B | C |
# |---|---|---|
# | 1 | 0 | 0 |
# | 0 | 1 | 0 |
# | 0 | 0 | 1 |
#
# -----------
#
# Here, the columns will be some relevants words that we believe to have predictive power.
#
# In order to find them, let's play arround with the data.

# %%
# The columns 'words' will contains a list of all the words in the 'clean' column
df['words'] = df.clean_review.str.split('\s+')
df.words[0][:10]

# %% [markdown]
# ## Feature engineering
# Now we will identify ALL the words that have been used and count how many time they have been used.
#
#
# I tried to use Counter from the collection package but found it to be really slow when I was passing it entire lists so I just decided to do it in my own way.

# %%
dict_count = {}
data = list(df.itertuples(index=False, name=None))
for d in data:
    for w in d[3]:
        if not w in dict_count:
            dict_count[w] = 1
        else:
            dict_count[w] +=1

df_count = pd.DataFrame(dict_count, index=['Count']).T.sort_values('Count')
df_count.tail()
# %% [markdown]
# ## Stop words problem
# And here we are, the famous stop words problems. 
# This was indeed pretty well expected, the words that are the most common will be completely useless in our case. 
#
# A good practice is to get rid of them.
# The sklearn library has a english stop word froze set, we will use it to do that
#
#

# %%
from sklearn.feature_extraction import stop_words
df_count = df_count.loc[~df_count.index.isin(stop_words.ENGLISH_STOP_WORDS)]
df_count.tail()

# %% [markdown]
# #### Next step
# Now for each of these words, I will add a column to the DataFrame and I want to count how many time each of them appear in each review. This is where we use "some kind" of one-hot-encoding technics. We will not populate with 1 or 0 but with a number of occurence. 

# %%
top_words = df_count.tail(1500).index.tolist()

# Rename the columns as their name might appear in the list of words
df = df.rename({
    "sentiment": '_predict',
    "review": "_review"
}, axis=1)

for word in top_words:
    df[word] = df.clean_review.str.count(word)

df.head()

# %% [markdown]
# ## Where are the interesting stuff ??
# Alright, now that we have counted everything, why don't we group our data by sentiment, positive or negative, and see if any words appears way more often in a group and not in the other

# %%
result = df[['_predict',*top_words]].groupby('_predict').mean().T
result['diff_'] = (result.negative / result.positive) -1
result.diff_.sort_values()

# %% [markdown]
# ### That is interesting
# So here we are, words such as beautiful and wonderful are wayyyyy more often used in a positive review than in a negative review. And words like worst, and awful are more often used in a negative review. 
#
# Again, I believe those results are pretty obvious, that is just common sense. However it still took us less time than coming up with 1500 words by yourself. 
#
# By looking at the data so far, I'm assuming that there should be some predictive power in our variable.

# %%
predict_df = df[['_predict', *result.index.to_list()]]

# %%
x = predict_df.drop('_predict', axis=1)
y = predict_df['_predict']
y = y.replace({'positive':1,'negative':0})
# %%
total = x.shape[0]
n = 25000
n_test = total - n
x_train = x.iloc[:n,:].values
y_train = y.iloc[:n].values

x_test =  x.iloc[n:,:].values
y_test =  y.iloc[n:].values

# %% [markdown]
# ## Standardize the data
# Some learning models require the data to be normalize in some way. 
# Here we will just standardize them.

# %%
x_train_std = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
x_test_std = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

# %% [markdown]
# # Learn
# It is time to create our model. 
#
# This problem is a classification problem. Therefore we can choose among the following learning technics :
#
# * Linear Models
#     * Logistic Regression
#     * Support Vector Machines
# * Nonlinear models
#     * K-nearest Neighbors (KNN)
#     * Kernel Support Vector Machines (SVM)
#     * Naïve Bayes
#     * Decision Tree Classification
#     * Random Forest Classification
#
# In order to evaluate the quality of our model we will be using the following metrics:
#
# * Accuracy: Correct Predictions / Total predictions
# * Precision: True Positive / (True Positive + False Positive)
# * Recall: True Positive / (True Positive + False Negative)

# %%
from sklearn.metrics import confusion_matrix,plot_confusion_matrix, accuracy_score, recall_score, precision_score
def scores(y, y_pred):
    precision = precision_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print('-----------------')
    print('Precision')
    print(precision)
    print('-----------------')
    print('Accuracy')
    print(accuracy)
    print('-----------------')
    print('Recall')
    print(recall)
    cnf_mat = confusion_matrix(y,y_pred)
    print('-----------------')
    print('Confusion Matrix')
    print(cnf_mat)


# %% [markdown]
# ## Logistic Regression 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
clf = LogisticRegression(C=0.7)
clf.fit(x_train_std, y_train)
y_pred = clf.predict(x_test_std)
scores(y_test, y_pred)

# %% [markdown]
# ## Random Forest

# %%
clf = RandomForestClassifier(max_depth=3,n_estimators=500, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
scores(y_test, y_pred)


