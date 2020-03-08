import sys
import pandas as pd
import sqlalchemy as db

def load_data(messages_filepath, categories_filepath):
    """
    Read message and categories CSV file from the given filepath.
    Perform ETL, merge into one dataframe and return.

    Parameter: messages_filepath (string) - File path for message CSV file
                categories_filepath (string) - File path for category CSV file
    Return: df (DataFrame) - DataFrame merging message and categories.
    """
    # Load message file
    # Set index to id column for convieneince of concat and merge across dataframe
    messages = pd.read_csv(messages_filepath)
    messages.set_index('id',inplace=True)

    # Load categories file
    categories = pd.read_csv(categories_filepath)
    # Set index to id column for convieneince of concat and merge across dataframe
    categories.set_index('id',inplace=True)

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # apply a lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda a : a[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract(r'([0-9])')
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """ Drop duplicate data and return only unique """
    return df.drop_duplicates(keep='first')

def save_data(df, database_filename):
    """ Save the DataFrame to sqlite DB file """

    # Check if database filename is valid
    if database_filename.find('.db')>=0:
        engine = db.create_engine('sqlite:///'+database_filename)
        df.to_sql('CategorisedMessages', engine, index=False)
    else:
        print('Invalid database filename (need name.db) : '+database_filename)

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