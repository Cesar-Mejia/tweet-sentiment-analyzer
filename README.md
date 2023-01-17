# Twitter Sentiment Analysis

This project is a machine learning platform that takes in twitter posts from a dataset and performs sentiment analysis on each post. The goal is to classify the sentiment of the tweets as positive or negative. 

## Dataset
The data set used features 1.6 million tweets extracted using the twitter API. The dataset can be viewed and downloaded at https://www.kaggle.com/datasets/kazanova/sentiment140

## Data Preprocessing
The data is preprocessed before it is classified. This includes removing special characters, numbers, and URLs, and converting all characters to lowercase. Stop words are also removed and lemmatization is performed on each word. 

## Feature Extraction
TF-IDF (term frequency-inverse document frequency) was used for feature extraction. This method is used to convert the textual data into numerical values that can be used as input for the classification algorithms.

## Classification Algorithms
The following classification algorithms were used in this project:
- NLTK's built-in sentiment analyzer
- Naive Bayes classifier
- Decision tree classifier
- Support Vector Machine (SVM)

## Results
The results were compared using the confusion matrices and accuracy scores of each classification algorithm. The overall performance of each algorithm was evaluated and the best-performing algorithm was chosen for the final model.


## Python Libraries
- NLTK
- Spacy
- scikit-learn
- pandas
- numpy
- Seaborn
- Matplotlib

## Conclusion
This project demonstrates the use of various machine learning algorithms for sentiment analysis on twitter data. The results of the different algorithms were compared and the best-performing algorithm was chosen for the final model. The project can be further improved by using more advanced preprocessing techniques and feature extraction methods.

