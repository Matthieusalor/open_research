# [markdown]
## Figuring out Natural Language Processing
## As I have never worker on NLP before, the purpose of this notebook was to playing arround with a dataset and trying to figure out a bunch of stuff on the subject.
## Here we will be working on the

# %%
import pandas as pd
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from collections import Counter

df = pd.read_csv('IMDB Dataset.csv')
df['clean'] = df['review'].str.replace('\d+', '')
df['clean'] = df['clean'].str.replace(r'[^A-Za-z0-9 ]+', '')
df['clean'] = df['clean'].str.lower()
df['clean'] = df['clean'].str.strip()
df['clean'] = df['clean'].str.replace('<[^<]+?>', '')

df['words'] = df.clean.str.split('\s+')
df = df.rename({"sentiment": '_predict'}, axis=1)
# %%

count = Counter()
dict_count = {}
data = list(df.itertuples(index=False, name=None))
for d in data:
    for w in d[3]:
        if not w in dict_count:
            dict_count[w] = 1
        else:
            dict_count[w] +=1

df_count = pd.DataFrame(dict_count, index=['Count']).T.sort_values('Count').tail(1500)
df.drop('review', inplace=True, axis=1)
# %%
top_words = df_count.index.tolist()

for word in top_words:
    df[word] = df.clean.str.count(word)

result = df[['_predict',*top_words]].groupby('_predict').mean().T
result['diff_'] = (result.negative / result.positive) -1
print(result.diff_.sort_values())

# %%
good_words = result.diff_.sort_values().head(750).index.tolist()
bad_words = result.diff_.sort_values().tail(750).index.tolist()


predict_df = df[['_predict', *good_words, *bad_words]]

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

# %%
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
from sklearn.metrics import confusion_matrix,plot_confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
clf = LogisticRegression(C=0.7)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
scores(y_test, y_pred)

# %%
clf = RandomForestClassifier(max_depth=3,n_estimators=500, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)
scores(y_train, y_pred)


