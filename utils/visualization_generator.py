import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from collections import Counter
import numpy as np

class VisualizationGenerator:
    def __init__(self):
        """Initialize the visualization generator."""
        self.sentiment_colors = {
            'Very Positive': '#00b894',
            'Positive': '#2ecc71',
            'Neutral': '#7f8c8d',
            'Negative': '#e74c3c',
            'Very Negative': '#c0392b'
        }
    
    def create_sentiment_pie_chart(self, df, sentiment_column='sentiment'):
        """
        Create a pie chart showing sentiment distribution.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            sentiment_column (str): Name of the sentiment column
            
        Returns:
            plotly.graph_objects.Figure: Pie chart figure
        """
        sentiment_counts = df[sentiment_column].value_counts()
        
        colors = [self.sentiment_colors.get(sentiment, '#95a5a6') 
                 for sentiment in sentiment_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title={
                'text': 'Sentiment Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            font=dict(size=14),
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_sentiment_bar_chart(self, df, sentiment_column='sentiment'):
        """
        Create a bar chart showing sentiment distribution.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            sentiment_column (str): Name of the sentiment column
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        sentiment_counts = df[sentiment_column].value_counts()
        
        colors = [self.sentiment_colors.get(sentiment, '#95a5a6') 
                 for sentiment in sentiment_counts.index]
        
        fig = go.Figure(data=[go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=colors,
            text=sentiment_counts.values,
            textposition='auto'
        )])
        
        fig.update_layout(
            title={
                'text': 'Sentiment Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Sentiment',
            yaxis_title='Count',
            font=dict(size=14),
            height=500
        )
        
        return fig
    
    def create_wordcloud(self, df, text_column, sentiment_filter=None):
        """
        Create a word cloud from text data.
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Name of the text column
            sentiment_filter (str): Filter by specific sentiment (optional)
            
        Returns:
            str: Base64 encoded image of the word cloud
        """
        # Filter by sentiment if specified
        if sentiment_filter:
            filtered_df = df[df['sentiment'] == sentiment_filter]
        else:
            filtered_df = df
        
        if filtered_df.empty:
            return None
        
        # Combine all text
        text = ' '.join(filtered_df[text_column].astype(str))
        
        if not text.strip():
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Convert to base64 image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud{" - " + sentiment_filter if sentiment_filter else ""}', 
                 fontsize=16, pad=20)
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
    def create_sentiment_timeline(self, df, date_column, sentiment_column='sentiment'):
        """
        Create a timeline showing sentiment trends over time.
        
        Args:
            df (pd.DataFrame): DataFrame with date and sentiment data
            date_column (str): Name of the date column
            sentiment_column (str): Name of the sentiment column
            
        Returns:
            plotly.graph_objects.Figure: Timeline figure
        """
        # Convert date column to datetime
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # Remove rows with invalid dates
        df_copy = df_copy.dropna(subset=[date_column])
        
        if df_copy.empty:
            return None
        
        # Group by date and sentiment
        timeline_data = df_copy.groupby([df_copy[date_column].dt.date, sentiment_column]).size().reset_index(name='count')
        timeline_data['date'] = pd.to_datetime(timeline_data[date_column])
        
        fig = px.line(
            timeline_data,
            x='date',
            y='count',
            color=sentiment_column,
            color_discrete_map=self.sentiment_colors,
            title='Sentiment Trends Over Time'
        )
        
        fig.update_layout(
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Date',
            yaxis_title='Count',
            font=dict(size=14),
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_confidence_distribution(self, df, confidence_column='confidence'):
        """
        Create a histogram showing confidence score distribution.
        
        Args:
            df (pd.DataFrame): DataFrame with confidence data
            confidence_column (str): Name of the confidence column
            
        Returns:
            plotly.graph_objects.Figure: Histogram figure
        """
        fig = px.histogram(
            df,
            x=confidence_column,
            nbins=20,
            title='Confidence Score Distribution',
            color_discrete_sequence=['#3498db']
        )
        
        fig.update_layout(
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            font=dict(size=14),
            height=500
        )
        
        return fig
    
    def create_sentiment_confidence_scatter(self, df, sentiment_column='sentiment', confidence_column='confidence'):
        """
        Create a scatter plot showing sentiment vs confidence.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment and confidence data
            sentiment_column (str): Name of the sentiment column
            confidence_column (str): Name of the confidence column
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure
        """
        fig = px.scatter(
            df,
            x=sentiment_column,
            y=confidence_column,
            color=sentiment_column,
            color_discrete_map=self.sentiment_colors,
            title='Sentiment vs Confidence Score'
        )
        
        fig.update_layout(
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Sentiment',
            yaxis_title='Confidence Score',
            font=dict(size=14),
            height=500
        )
        
        return fig
    
    def create_summary_stats_table(self, df, sentiment_column='sentiment', confidence_column='confidence'):
        """
        Create a summary statistics table.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            sentiment_column (str): Name of the sentiment column
            confidence_column (str): Name of the confidence column
            
        Returns:
            dict: Summary statistics
        """
        stats = {
            'Total Entries': len(df),
            'Unique Sentiments': df[sentiment_column].nunique(),
            'Most Common Sentiment': df[sentiment_column].mode().iloc[0] if not df.empty else 'N/A',
            'Average Confidence': f"{df[confidence_column].mean():.2f}" if confidence_column in df.columns else 'N/A',
            'High Confidence (>0.8)': len(df[df[confidence_column] > 0.8]) if confidence_column in df.columns else 'N/A'
        }
        
        return stats