import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class AdvAdjExtractor(BaseEstimator, TransformerMixin):
    """
    This is the custom transformer class.
    This class will add additional features to the data by counting the number of adverb and adjectives
    add to the input dataframe and return the result.
    """

    def adj_adv_count(self, text):
        """
        This function expected input as sentence.
        The function will tokenize and lemmatize the token then add PoS tag to each token.
        Finally, return the count of adverb and adjectives.

        Parameter: text (string) - sentence to count adj and adv
        Return: adj_adv (int) - count of adj_adv
        """
        # Tokenize and clean with lemmatizer
        token_list = self.tokenize(text)
        # PoS tag to each token
        pos_tags = nltk.pos_tag(token_list)
        # Initialize the counter
        adj_adv = 0
        # Iterate thru the token list and count the adverb or adjectives
        for word, tag in pos_tags:
            if tag in ['JJ','JJR','JJS','RB','RBR','RBS']:
                adj_adv = adj_adv+1
        return adj_adv

    def tokenize(self,text):
        """
        Tokenize the sentence and also clean the token with lemmatizer.

        Parameter: text (string) - Sentence to be tokenize
        Return: clean_tokens (list) - List of lemmatized token
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

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """ Apply adverb and adjective count to all elements """
        X_tagged = pd.Series(X).apply(self.adj_adv_count)
        return pd.DataFrame(X_tagged)
