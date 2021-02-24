# import libraries
import sys
import numpy as np
import re
import pickle
import pandas as pd 
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.base import BaseEstimator,TransformerMixin

import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

nltk.download(['wordnet', 'punkt', 'stopwords'])


def load_data(database_filepath):
    """
    Function:
       load data from database
    Args:
       database_filepath: the path of the database
    Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('tb_disaster_messages', engine)
    
    # message Column
    X = df['message']  
    # classification label
    Y = df.iloc[:, 4:] 

    return X, Y

def tokenize(text):
    """
    Function: 
      split text into words and return the root form of the words
    Args:
      text(str): the message
    Return:
      clean_tokens(list of str): a list of the root form of the message words
    """

    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    

def build_model():
    """
     Function: 
        build a model for classifing the disaster messages
     Return:
        cv(list of str): classification model
     """
   
    # pipeline: Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    

    # Create Grid search parameters for Random Forest Classifier   
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    Function to generate classification report on the model
    Input: 
        model, test set: X_test & Y_test
    Output: 
        Prints the classification report
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Function: Save a pickle file of the model
    Input:
        model: the classification model
        model_filepath (str): the path of pickle file
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()