from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        sentiment_score = torch.argmax(probabilities).item() + 1
        
        if sentiment_score == 1:
            return "Very Negative"
        elif sentiment_score == 2:
            return "Negative"
        elif sentiment_score == 3:
            return "Neutral"
        elif sentiment_score == 4:
            return "Positive"
        else:
            return "Very Positive"