from flask import Flask, render_template, request, send_file
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer
import io

app = Flask(__name__)
sentiment_analyzer = SentimentAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                if 'feedback' in df.columns:
                    df['sentiment'] = df['feedback'].apply(sentiment_analyzer.analyze_sentiment)
                    output = io.BytesIO()
                    output_filename = f"{file.filename.split('.')[0]}_sentiments.csv"
                    df.to_csv(output, index=False, encoding='utf-8')
                    output.seek(0)
                    return send_file(output, mimetype='text/csv', as_attachment=True, download_name=output_filename)
                else:
                    return render_template('index.html', error='CSV file must contain a "feedback" column.')
            else:
                return render_template('index.html', error='Please upload a CSV file.')
        elif 'text' in request.form:
            text = request.form['text']
            sentiment = sentiment_analyzer.analyze_sentiment(text)
            return render_template('result.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)