import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])

import sys
import re
import pandas as pd
import sqlalchemy as db
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from distributions import AdvAdjExtractor

def load_data(database_filepath):
    """
    Read data from database and separated to set of the input and classification categories 
    to train and evaluate the model.
    
    Parameter: dabase filepath e.g. data.db
    Output: X as list of input message, Y as list of categories and list of categories name
    
    """
    engine = db.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('CategorisedMessages', engine)
    # List of categories of the db
    category_names = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    # Input to classify
    X = df['message']
    # Classification result to train the ML
    Y = df[category_names]

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the sentence and also clean the token with lemmatizer.
    
    Input: Sentence to be tokenize
    Output: List of lemmatized token
    
    """
    # Replace none-character with space
    text = re.sub('[^A-Za-z0-9]',' ',text)

    # Tokenize the input
    tokens = word_tokenize(text)

    # Initialize lemmatizer for standardize form of words
    lemmatizer = WordNetLemmatizer()

    # We will iterate each token in the list, lemmatize and return result
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # The pipeline will use Count Vectorize, TFIDF 
    # and number of adverb and adjective which is a custom function to add new features to train the ML
    # The best model classification model as tried so far is Random Forest.
    pipeline = Pipeline([
        ('features',FeatureUnion([
            ('text_pipeline',Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('adv_adj', AdvAdjExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameter for GridSearch, since the data size and categories are quite large the list below balance the training time and result
    parameters = {
        'clf__estimator__n_estimators': [10,20]
    }

    return GridSearchCV(pipeline, param_grid=parameters,n_jobs=4)


def evaluate_model(model, X_test, Y_test, category_names):
    # Predict the test data set
    y_pred = model.predict(X_test)

    # Convert the prediction result to dataframe for ease of iteration and comparison
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred.columns=category_names

    # Evaluate improvement result
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column],df_y_pred[column]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

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