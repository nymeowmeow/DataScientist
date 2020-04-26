# import libraries
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from sqlalchemy import create_engine
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(db_path):
    """
        load data from sqlite db to dataframe, and returns features, target and label name
        Args:
            db_path : path to sqlite db
        Returns:
            X       - features
            Y       - target
            Y_label - label of target columns
    """
    #load data from sqlite db to dataframe
    engine = create_engine('sqlite:///{}'.format(db_path))
    df = pd.read_sql_table('messages', engine)
    #extract features, target and target labels
    X = df['message'].values
    Y_df = df.drop(columns=['message', 'id', 'genre', 'original'])
    Y = Y_df.values
    Y_labels = Y_df.columns

    return X, Y, Y_labels

def build_model():
    """
        build machine learning model
        Returns:
            model - machine learning pipeline for the classification problem
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    parameters = {
        #'vectorizer__max_df': (0.75, 1.0),
        #'tfidf__ngram_range': ((1, 1), (1, 3)),
        #'vectorizer__max_features': (None, 8000),
        #'clf__estimator__n_estimators' : [50, 100, 250 ],
        'clf__estimator__n_estimators' : [100, 200],
        #'clf__estimator__min_samples_split': [2, 10, 50]
    }
    model = GridSearchCV(pipeline, param_grid=parameters,cv=3,n_jobs=-1,verbose=1)

    return model

def tokenize(text):
    """
        tokenized the text, translated to lower case, remove punctuation using regular expression,
        remove stopwords and apply lemmatization and stemming
        Args:
            text - text to be tokenized
        Returns:
            list - list of tokens extracted from input
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    lemmatizer = WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()

    tokens = [ word for word in word_tokenize(text) if word not in stopwords.words('english') ]
    return [ stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens ]

def evaluate_model(model, X_test, Y_test, Y_labels):
    """
        evaluate the performance of the model, in terms of F1 score etc that is available from
        the classification report and print the result to the console
        Args:
            model - trained model to be evaluated
            X_test - testing data
            Y_test - testing target value
            Y_labels - labels of target categories
        Returns:
            None 
    """
    y_pred = model.predict(X_test)
    print (classification_report(Y_test, y_pred, target_names=Y_labels))

def save_model(model, model_path):
    """
       saves the model to the model path provided as pickle file
       Args:
           model - model to be saved
           model_path - path where the model should be saved as pickle file
       Returns:
           None
    """
    model_pkl = open(model_path, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()

def main():
    if len(sys.argv) == 3:
        db_path, model_path = sys.argv[1:]
        print (f'loading data from {db_path} to dataframe')
        X, Y, Y_labels = load_data(db_path)
        #split the data into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
        #build model
        model = build_model()
        #train model
        print ('training model....')
        model.fit(X_train, y_train)
        #evaluate the performance of the model
        evaluate_model(model, X_test, y_test, Y_labels)
        #save model to the model path name provided, in pkl format
        print ('saving model to {}'.format(model_path))
        save_model(model, model_path)
    else:
        print ("Usage: python train_classifier.py <path-to-sqlite-db> <outputmodel-path>")

if __name__ == '__main__':
    main()
