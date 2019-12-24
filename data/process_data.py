import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df


def clean_data(df):
    categories_split = df["categories"].str.split(";", expand=True)
    category_colnames = list(categories_split.loc[0].str.rsplit("-").apply(lambda x: x[0]))
    categories_split.columns = category_colnames
    
    num_extract = lambda x: int(x.split("-")[-1])
    categories_split = categories_split.applymap(num_extract).astype(dtype=int)
    
    try:
        df = df.drop(list(categories.columns), axis=1)

    except:
        pass
    
    df = pd.concat([df, categories_split], axis=1).drop_duplicates()
 
    return df


def save_data(df, database_filename):
    engine = create_engine("""sqlite:///""" + database_filename)
    df.to_sql("DisasterResponse", con=engine, if_exists="replace")
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.columns)
        
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