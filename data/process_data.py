#import libraries
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(messages_filepath, categories_filepath):
    """
    - Takes two CSV files as input
    - Imports them as pandas dataframe.
    - Merges them into a single dataframe
    
    Args:
    messages_file_path (str): Messages CSV file
    categories_file_path (str): Categories CSV file
    
    Returns:
    df (pandas_dataframe): Dataframe obtained from merging the two input\
    data
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    Function to clean the combined dataframe to be used by ML model
    
    Args:
    df (pandas_dataframe): Merged dataframe returned from load_data() function
    
    Returns:
    df (pandas_dataframe): Cleaned data to be used by ML model
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)  
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    category_colnames = []
    for i in row:
        category_colnames.append(i[:-2])   
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors = 'coerce')
    
    # Replace categories column in df with new category columns.
    
    # drop the original categories column from `df`
    df.drop (['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, sort = False)
    
    # drop duplicates
    df = df.drop_duplicates(subset = ['message'])
    
    return df
    


def save_data(df, database_filename):
    """
    Function to save cleaned data to an SQL database
    
    Args:
    df (pandas_dataframe): Cleaned data returned from clean_data() function
    database_file_name (str): File path of SQL Database into which the cleaned\
    data is to be saved
    
    Returns:
    None
    """

    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('messages_categories', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()