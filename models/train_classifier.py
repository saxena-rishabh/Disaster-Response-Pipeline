# import libraries
import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import re
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.metrics import fbeta_score, make_scorer
import pickle

import matplotlib.pyplot as plt

from sqlalchemy import create_engine
import sys
import warnings
warnings.filterwarnings('ignore')

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Function to load data from SQL Database
    
    Args:
    database_filepath: SQL database file
    
    Returns:
    X (pandas_dataframe): Features dataframe
    Y (pandas_dataframe): Target dataframe
    category_names (list): Target labels 
    """
        
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', con = engine)
    X = df ['message']
    y = df.iloc[:,4:]
    
    # Y['related'] contains three distinct values
    # mapping extra values to `1`
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
    Function to tokenize text data
    
    Args:
    text (str): Messages as text data
    
    Returns:
    words (list): Processed text after normalizing, tokenizing and lemmatizing
    """
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function to build model based on best parameters
    predetermined by grid search during development phase.
    
    Returns:
    Machine Learning Pipeline
    """
    
    pipeline_ada = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(AdaBoostClassifier()))
        ])
        
    parameters_ada =  {'clf__estimator__learning_rate': [0.2, 0.5, 1],
              'clf__estimator__n_estimators': [50, 100]}

    cv = GridSearchCV(pipeline_ada, param_grid=parameters_ada, verbose=3)
    
    return cv
    
    ''' WITHOUT GRID SEARCH
    pipeline_final = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', multioutput.MultiOutputClassifier(AdaBoostClassifier(algorithm='SAMME.R',
                                                                        base_estimator=None,
                                                                        learning_rate=0.5,
                                                                        n_estimators=100,
                                                                        random_state=None)))
            ])
    
    return pipeline_final
    '''


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    # Build classification report
    print(classification_report(Y_test, Y_pred,
                                target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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