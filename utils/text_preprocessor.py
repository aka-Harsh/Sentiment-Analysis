import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with required NLTK data."""
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_data(self):
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)
    
    def clean_text(self, text, apply_preprocessing=True):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Input text to clean
            apply_preprocessing (bool): Whether to apply full preprocessing
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove emojis and special unicode characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not apply_preprocessing:
            return text
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and single characters
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column, apply_preprocessing=True):
        """
        Preprocess text data in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column to preprocess
            apply_preprocessing (bool): Whether to apply full preprocessing
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text
        """
        df_copy = df.copy()
        
        # Clean the text column
        df_copy[f'{text_column}_cleaned'] = df_copy[text_column].apply(
            lambda x: self.clean_text(x, apply_preprocessing)
        )
        
        # Remove empty rows after cleaning
        df_copy = df_copy[df_copy[f'{text_column}_cleaned'].str.len() > 0]
        
        return df_copy
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text using TextBlob.
        
        Args:
            text (str): Input text
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of top keywords
        """
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        blob = TextBlob(text)
        
        # Get noun phrases and filter
        noun_phrases = [phrase.lower() for phrase in blob.noun_phrases 
                       if len(phrase.split()) <= 3 and len(phrase) > 2]
        
        # Count frequency
        word_freq = {}
        for phrase in noun_phrases:
            word_freq[phrase] = word_freq.get(phrase, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [keyword for keyword, freq in sorted_keywords[:top_n]]