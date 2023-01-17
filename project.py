from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import spacy
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# loads in dataset into dataframe
header = ['target', 'ids', 'date', 'flag', 'user', 'text']
data = pd.read_csv('tweets.csv', header=None, names=header)

# drop the unneeded columns from dataset
data.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)
print(data)


# reduce the number of samples in the dataset
x = data.iloc[:, 1].values
y = data.iloc[:, 0].values

x, _, y, _ = train_test_split(x, y, test_size=0.99, random_state=0)


# loading in english library
nlp = spacy.load("en_core_web_sm")


# function preprocesses text data
def preprocessing(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"@[A-Za-z0-9]+", ' ', sentence)
    sentence = re.sub(r"https?://[A-Za-z0-9./]+", ' ', sentence)

    tokens = [
        token.text for token in nlp(sentence) if not (
            token.is_stop or
            token.like_num or
            token.is_punct or
            token.is_space or
            len(token) == 1 or
            '@' in token.text or not
            token.is_alpha
        )
    ]
    tokens = ' '.join([element for element in tokens])

    # lemmatization
    lemmas = []
    for token in nlp(tokens):
        lemmas.append(token.lemma_)
    lemmas = ' '.join([element for element in lemmas])
    return lemmas


# preprocess text data
x_clean = [preprocessing(text) for text in x]

# preprocessed data dataframe
data_clean = pd.DataFrame({
    "Tweet": x_clean,
    "Sentiment": y
})

# drop all rows with empty text
data_clean = data_clean[data_clean['Tweet'] != '']
print(data_clean)

# split clean data into a train/test data
X_train, X_test, y_train, y_test = train_test_split(
    data_clean['Tweet'], data_clean['Sentiment'], test_size=0.3, random_state=0)

print(pd.DataFrame(X_train))

classifiers = []
accuracy_scores = []

# helper function to print performance statistics


def printStats(y_test, predictions):
    print('Confusion Matrix\n')
    print(confusion_matrix(y_test, predictions))

    print('\n\nAccuracy Score:', round(accuracy_score(y_test, predictions), 2))


# classification using NLTK
sa = SentimentIntensityAnalyzer()

data_clean['predictions'] = data_clean['Tweet'].apply(
    lambda tweet: sa.polarity_scores(tweet)['compound'])
data_clean['predictions'] = data_clean['predictions'].apply(
    lambda pred: 4 if pred >= 0 else 0)


printStats(data_clean['Sentiment'], data_clean['predictions'])

accuracy_scores.append(round(accuracy_score(
    data_clean['Sentiment'], data_clean['predictions']), 2))
classifiers.append('NLTK')


# create pipeline and fit training data to model
pipeline_SVC = Pipeline([('tfidf',  TfidfVectorizer()), ('classifier', SVC())])
pipeline_SVC.fit(X_train, y_train)


# record predictions
predictions_SVC = pipeline_SVC.predict(X_test)

printStats(y_test, predictions_SVC)

accuracy_scores.append(round(accuracy_score(y_test, predictions_SVC), 2))
classifiers.append('Support Vector')


# Decision Tree Classifier
pipeline_DTC = Pipeline([('tfidf', TfidfVectorizer()),
                        ('classifier', DecisionTreeClassifier())])
pipeline_DTC.fit(X_train, y_train)

predictions_DTC = pipeline_DTC.predict(X_test)

printStats(y_test, predictions_DTC)

accuracy_scores.append(round(accuracy_score(y_test, predictions_DTC), 2))
classifiers.append('Decision Tree')


# Naive Bayes Classifier
pipeline_NBC = Pipeline([('tfidf', TfidfVectorizer()),
                        ('classifier', MultinomialNB())])
pipeline_NBC.fit(X_train, y_train)

predictions_NBC = pipeline_NBC.predict(X_test)

printStats(y_test, predictions_NBC)

accuracy_scores.append(round(accuracy_score(y_test, predictions_NBC), 2))
classifiers.append('Naive Bayes')

scores = np.array(accuracy_scores)*100
scores.sort()
print(classifiers)
print(scores)


plt.figure(dpi=100)
classifiers = ['NLTK', 'Decision Tree', 'Naive Bayes', 'Support Vector']
sns.lineplot(x=classifiers, y=scores)
plt.ylabel('Accuracy (%)')
plt.title('Classifier Accuracies')
plt.show()
