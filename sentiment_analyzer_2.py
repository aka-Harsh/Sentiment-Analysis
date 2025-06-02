from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with BERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_labels = {
            1: "Very Negative",
            2: "Negative", 
            3: "Neutral",
            4: "Positive",
            5: "Very Positive"
        }
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            str: Sentiment label
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return "Neutral"
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        sentiment_score = torch.argmax(probabilities).item() + 1
        
        return self.sentiment_labels[sentiment_score]
    
    def analyze_sentiment_with_confidence(self, text):
        """
        Analyze sentiment with confidence score.
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            tuple: (sentiment_label, confidence_score)
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return "Neutral", 0.0
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence_score = torch.max(probabilities).item()
        sentiment_score = torch.argmax(probabilities).item() + 1
        
        return self.sentiment_labels[sentiment_score], round(confidence_score, 3)
    
    def analyze_batch(self, texts, batch_size=32, show_progress=True):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of texts to analyze
            batch_size (int): Number of texts to process at once
            show_progress (bool): Whether to show progress bar
            
        Returns:
            list: List of tuples (sentiment, confidence)
        """
        results = []
        
        # Process in batches
        if show_progress:
            iterator = tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment")
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                sentiment, confidence = self.analyze_sentiment_with_confidence(text)
                batch_results.append((sentiment, confidence))
            
            results.extend(batch_results)
        
        return results
    
    def analyze_dataframe(self, df, text_column, show_progress=True):
        """
        Analyze sentiment for texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text
            show_progress (bool): Whether to show progress bar
            
        Returns:
            pd.DataFrame: DataFrame with sentiment and confidence columns added
        """
        df_copy = df.copy()
        
        # Get texts as list
        texts = df_copy[text_column].astype(str).tolist()
        
        # Analyze sentiment
        results = self.analyze_batch(texts, show_progress=show_progress)
        
        # Add results to DataFrame
        sentiments, confidences = zip(*results)
        df_copy['sentiment'] = sentiments
        df_copy['confidence'] = confidences
        
        return df_copy
    
    def get_sentiment_distribution(self, df, sentiment_column='sentiment'):
        """
        Get sentiment distribution statistics.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            sentiment_column (str): Name of the sentiment column
            
        Returns:
            dict: Sentiment distribution statistics
        """
        if sentiment_column not in df.columns:
            return {}
        
        distribution = df[sentiment_column].value_counts().to_dict()
        total_count = len(df)
        
        # Calculate percentages
        distribution_pct = {
            sentiment: f"{count} ({count/total_count*100:.1f}%)" 
            for sentiment, count in distribution.items()
        }
        
        return distribution_pct
    
    def get_confidence_stats(self, df, confidence_column='confidence'):
        """
        Get confidence score statistics.
        
        Args:
            df (pd.DataFrame): DataFrame with confidence data
            confidence_column (str): Name of the confidence column
            
        Returns:
            dict: Confidence statistics
        """
        if confidence_column not in df.columns:
            return {}
        
        confidence_scores = df[confidence_column]
        
        stats = {
            'Mean Confidence': round(confidence_scores.mean(), 3),
            'Median Confidence': round(confidence_scores.median(), 3),
            'Min Confidence': round(confidence_scores.min(), 3),
            'Max Confidence': round(confidence_scores.max(), 3),
            'High Confidence (>0.8)': len(confidence_scores[confidence_scores > 0.8]),
            'Low Confidence (<0.5)': len(confidence_scores[confidence_scores < 0.5])
        }
        
        return stats
