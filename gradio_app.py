import gradio as gr
import pandas as pd
import numpy as np
from sentiment_analyzer_2 import SentimentAnalyzer
from utils.text_preprocessor import TextPreprocessor
from utils.visualization_generator import VisualizationGenerator
import plotly.graph_objects as go
import base64
from io import BytesIO
from PIL import Image
import os
import tempfile

class SentimentAnalysisApp:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_preprocessor = TextPreprocessor()
        self.viz_generator = VisualizationGenerator()
        self.processed_data = None
    
    def analyze_single_text(self, text, apply_preprocessing):
        if not text or not text.strip():
            return "⚠️ Please enter some text to analyze.", "", ""
        
        if apply_preprocessing:
            cleaned_text = self.text_preprocessor.clean_text(text, apply_preprocessing=True)
            analysis_text = cleaned_text
        else:
            cleaned_text = text
            analysis_text = text
        
        sentiment, confidence = self.sentiment_analyzer.analyze_sentiment_with_confidence(analysis_text)
        
        sentiment_emoji = {
            "Very Positive": "😍",
            "Positive": "😊", 
            "Neutral": "😐",
            "Negative": "😔",
            "Very Negative": "😢"
        }
        
        emoji = sentiment_emoji.get(sentiment, "🤔")
        result = f"## {emoji} Sentiment Analysis Result\n\n**Sentiment:** {sentiment}\n\n**Confidence:** {confidence:.3f} ({confidence*100:.1f}%)"
        
        confidence_bar = f"Confidence: {confidence:.1%}"
        
        return result, cleaned_text, confidence_bar
    
    def process_csv_file(self, file, text_column, apply_preprocessing, progress=gr.Progress()):
        if file is None:
            return "⚠️ Please upload a CSV file.", None, gr.update(choices=[], value=None)
        
        try:
            progress(0.1, desc="📖 Reading CSV file...")
            df = pd.read_csv(file.name)
            
            if df.empty:
                return "❌ The uploaded file is empty.", None, gr.update(choices=[], value=None)
            
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            if not text_column or text_column not in df.columns:
                if text_columns:
                    return f"⚠️ Please select a valid text column. Available: {', '.join(text_columns)}", None, gr.update(choices=text_columns, value=text_columns[0])
                else:
                    return "❌ No text columns found in the CSV file.", None, gr.update(choices=[], value=None)
            
            progress(0.2, desc="🧹 Preprocessing text...")
            
            if apply_preprocessing:
                df = self.text_preprocessor.preprocess_dataframe(df, text_column, apply_preprocessing=True)
                analysis_column = f'{text_column}_cleaned'
            else:
                analysis_column = text_column
            
            progress(0.4, desc="🤖 Analyzing sentiment...")
            
            df = self.sentiment_analyzer.analyze_dataframe(df, analysis_column, show_progress=False)
            
            progress(0.8, desc="📊 Generating summary...")
            
            self.processed_data = df
            
            total_entries = len(df)
            sentiment_dist = self.sentiment_analyzer.get_sentiment_distribution(df)
            confidence_stats = self.sentiment_analyzer.get_confidence_stats(df)
            
            summary_html = self._create_animated_summary(total_entries, sentiment_dist, confidence_stats)
            
            progress(0.9, desc="💾 Preparing download...")
            output_file = self._prepare_download_file(df, file.name)
            
            progress(1.0, desc="✅ Complete!")
            
            return summary_html, output_file, gr.update(choices=text_columns, value=text_column)
            
        except Exception as e:
            return f"❌ Error processing file: {str(e)}", None, gr.update(choices=[], value=None)
    
    def _create_animated_summary(self, total_entries, sentiment_dist, confidence_stats):
        html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 20px;
            color: white;
            margin: 20px 0;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            animation: slideIn 0.8s ease-out;
        ">
            <style>
                @keyframes slideIn {{
                    from {{ opacity: 0; transform: translateY(-20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.05); }}
                }}
                @keyframes float {{
                    0%, 100% {{ transform: translateY(0px); }}
                    50% {{ transform: translateY(-10px); }}
                }}
                .stat-card {{
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    margin: 10px;
                    animation: float 3s ease-in-out infinite;
                    transition: transform 0.3s ease;
                }}
                .stat-card:hover {{
                    transform: translateY(-5px) scale(1.02);
                    background: rgba(255,255,255,0.2);
                }}
                .big-number {{
                    font-size: 3em;
                    font-weight: bold;
                    animation: pulse 2s infinite;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .progress-bar {{
                    width: 100%;
                    height: 8px;
                    background: rgba(255,255,255,0.2);
                    border-radius: 4px;
                    overflow: hidden;
                    margin: 10px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #00f5ff, #0099ff);
                    border-radius: 4px;
                    animation: slideProgress 2s ease-out;
                }}
                @keyframes slideProgress {{
                    from {{ width: 0%; }}
                    to {{ width: 100%; }}
                }}
            </style>
            
            <h2 style="text-align: center; margin-bottom: 30px; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🎯 Analysis Complete!
            </h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div class="stat-card">
                    <div style="text-align: center;">
                        <div class="big-number">{total_entries}</div>
                        <p style="margin: 0; font-size: 1.2em;">📊 Total Entries</p>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div style="text-align: center;">
                        <div class="big-number">{confidence_stats.get('Mean Confidence', 'N/A')}</div>
                        <p style="margin: 0; font-size: 1.2em;">🎯 Avg Confidence</p>
                    </div>
                </div>
                
                <div class="stat-card">
                    <div style="text-align: center;">
                        <div class="big-number">{confidence_stats.get('High Confidence (>0.8)', 'N/A')}</div>
                        <p style="margin: 0; font-size: 1.2em;">⭐ High Confidence</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div style="
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 25px; 
            margin: 25px 0;
            animation: slideIn 1s ease-out 0.3s both;
        ">
            <div style="
                background: linear-gradient(145deg, #f8f9fa, #e9ecef);
                padding: 25px;
                border-radius: 20px;
                border-left: 6px solid #28a745;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            ">
                <h3 style="color: #495057; margin-top: 0; font-size: 1.5em;">🎭 Sentiment Distribution</h3>"""
        
        for sentiment, count in sentiment_dist.items():
            html += f"""
                <div style="margin: 15px 0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #6c757d;">{sentiment}:</span>
                    <span style="
                        background: linear-gradient(90deg, #007bff, #0056b3);
                        color: white;
                        padding: 5px 15px;
                        border-radius: 20px;
                        font-weight: bold;
                    ">{count}</span>
                </div>"""
        
        html += """
            </div>
            
            <div style="
                background: linear-gradient(145deg, #f8f9fa, #e9ecef);
                padding: 25px;
                border-radius: 20px;
                border-left: 6px solid #007bff;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            ">
                <h3 style="color: #495057; margin-top: 0; font-size: 1.5em;">📈 Confidence Statistics</h3>"""
        
        for stat, value in confidence_stats.items():
            html += f"""
                <div style="margin: 15px 0; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #6c757d;">{stat}:</span>
                    <span style="
                        background: linear-gradient(90deg, #28a745, #20c997);
                        color: white;
                        padding: 5px 15px;
                        border-radius: 20px;
                        font-weight: bold;
                    ">{value}</span>
                </div>"""
        
        html += """
            </div>
        </div>
        
        <div style="
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            border-radius: 15px;
            color: white;
            font-size: 1.2em;
            font-weight: bold;
            animation: pulse 2s infinite;
        ">
            🎉 Analysis completed successfully! Check the Visualizations tab for detailed insights.
        </div>
        """
        
        return html
    
    def _prepare_download_file(self, df, original_filename):
        import tempfile
        
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        output_filename = f"{base_name}_sentiment_analysis.csv"
        
        # Create a temporary file in the system temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, output_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Save the file
        df.to_csv(temp_path, index=False, encoding='utf-8')
        
        return temp_path
    
    def generate_visualizations(self):
        if self.processed_data is None:
            return None, None, None, None, None, "⚠️ No data to visualize. Please process a CSV file first."
        
        try:
            df = self.processed_data
            
            pie_chart = self.viz_generator.create_sentiment_pie_chart(df)
            bar_chart = self.viz_generator.create_sentiment_bar_chart(df)
            confidence_chart = self.viz_generator.create_confidence_distribution(df)
            
            wordcloud_img = None
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'comment' in col.lower() or 'review' in col.lower() or 'feedback' in col.lower() or 'cleaned' in col.lower()]
            
            if text_columns:
                wordcloud_base64 = self.viz_generator.create_wordcloud(df, text_columns[0])
                if wordcloud_base64:
                    img_data = base64.b64decode(wordcloud_base64)
                    wordcloud_img = Image.open(BytesIO(img_data))
            
            timeline_chart = None
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                timeline_chart = self.viz_generator.create_sentiment_timeline(df, date_columns[0])
            
            return pie_chart, bar_chart, confidence_chart, wordcloud_img, timeline_chart, "✅ Visualizations generated successfully! 🎨"
            
        except Exception as e:
            return None, None, None, None, None, f"❌ Error generating visualizations: {str(e)}"
    
    def _get_csv_columns(self, file):
        if file is None:
            return gr.update(choices=[], value=None)
        
        try:
            df = pd.read_csv(file.name, nrows=5)
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if text_columns:
                return gr.update(choices=text_columns, value=text_columns[0])
            else:
                return gr.update(choices=[], value=None)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return gr.update(choices=[], value=None)
    
    def create_interface(self):
        custom_css = """
        .gradio-container {
            max-width: 1400px !important;
            margin: auto;
        }
        
        .tab-nav button {
            font-size: 18px !important;
            font-weight: 600 !important;
            padding: 15px 25px !important;
            margin: 5px !important;
            border-radius: 15px !important;
            transition: all 0.3s ease !important;
            background: linear-gradient(145deg, #f8f9fa, #e9ecef) !important;
            border: none !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
        }
        
        .tab-nav button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2) !important;
            background: linear-gradient(145deg, #007bff, #0056b3) !important;
            color: white !important;
        }
        
        .tab-nav button.selected {
            background: linear-gradient(145deg, #28a745, #20c997) !important;
            color: white !important;
            transform: translateY(-2px) !important;
        }
        
        .gr-button {
            background: linear-gradient(145deg, #007bff, #0056b3) !important;
            border: none !important;
            border-radius: 15px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            padding: 12px 30px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 5px 15px rgba(0,123,255,0.3) !important;
        }
        
        .gr-button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 25px rgba(0,123,255,0.4) !important;
            background: linear-gradient(145deg, #0056b3, #004085) !important;
        }
        
        .gr-textbox, .gr-dropdown {
            border-radius: 15px !important;
            border: 2px solid #e9ecef !important;
            transition: all 0.3s ease !important;
        }
        
        .gr-textbox:focus, .gr-dropdown:focus {
            border-color: #007bff !important;
            box-shadow: 0 0 20px rgba(0,123,255,0.2) !important;
        }
        
        .gr-file {
            border: 3px dashed #007bff !important;
            border-radius: 20px !important;
            background: linear-gradient(145deg, #f8f9ff, #e6f3ff) !important;
            transition: all 0.3s ease !important;
        }
        
        .gr-file:hover {
            border-color: #0056b3 !important;
            background: linear-gradient(145deg, #e6f3ff, #cce7ff) !important;
            transform: scale(1.02) !important;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .gr-row, .gr-column {
            animation: fadeInUp 0.6s ease-out !important;
        }
        
        .gr-plot {
            border-radius: 15px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
        }
        """
        
        with gr.Blocks(
            title="🎭 Advanced Sentiment Analysis Tool",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="green",
                neutral_hue="slate",
                font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
            ),
            css=custom_css
        ) as interface:
            
            gr.HTML("""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 30px; color: white; animation: fadeInUp 0.8s ease-out;">
                    <h1 style="font-size: 3.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎭 Advanced Sentiment Analysis</h1>
                    <p style="font-size: 1.3em; margin: 15px 0 0 0; opacity: 0.9;">Analyze text sentiment with AI-powered insights, beautiful visualizations, and advanced preprocessing</p>
                    <div style="margin-top: 20px;">
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 10px;">🤖 AI-Powered</span>
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 10px;">📊 Interactive Charts</span>
                        <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 10px;">🎨 Beautiful UI</span>
                    </div>
                </div>
            """)
            
            with gr.Tabs():
                with gr.Tab("📝 Single Text Analysis", elem_id="single-text-tab"):
                    gr.HTML("<h2 style='text-align: center; color: #495057; margin-bottom: 20px;'>🔍 Analyze Individual Text</h2>")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_input = gr.Textbox(
                                label="✍️ Enter your text",
                                placeholder="Type or paste your text here... (reviews, comments, feedback, etc.)",
                                lines=6,
                                max_lines=10
                            )
                            
                            single_preprocessing = gr.Checkbox(
                                label="🧹 Apply text preprocessing (recommended)",
                                value=True,
                                info="Removes URLs, special characters, applies lemmatization for better accuracy"
                            )
                            
                            analyze_btn = gr.Button(
                                "🔍 Analyze Sentiment",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            sentiment_result = gr.Markdown(
                                label="📊 Analysis Result",
                                value="👋 Enter some text and click analyze to get started!"
                            )
                            confidence_display = gr.Textbox(
                                label="🎯 Confidence Level",
                                interactive=False
                            )
                            cleaned_text_display = gr.Textbox(
                                label="🧹 Processed Text",
                                interactive=False,
                                lines=4,
                                info="Shows cleaned text if preprocessing is applied"
                            )
                
                with gr.Tab("📊 CSV File Analysis", elem_id="csv-tab"):
                    gr.HTML("<h2 style='text-align: center; color: #495057; margin-bottom: 20px;'>📁 Batch Analysis from CSV</h2>")
                    
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(
                                label="📂 Upload CSV File (Drag & Drop Supported)",
                                file_types=[".csv"],
                                type="filepath",
                                height=120
                            )
                            
                            column_dropdown = gr.Dropdown(
                                label="🎯 Select Text Column",
                                choices=[],
                                value=None,
                                interactive=True,
                                info="Choose the column containing text to analyze (auto-detected)",
                                allow_custom_value=False
                            )
                            
                            csv_preprocessing = gr.Checkbox(
                                label="🧹 Apply text preprocessing",
                                value=True,
                                info="Recommended for better analysis accuracy"
                            )
                            
                            process_btn = gr.Button(
                                "🚀 Process CSV File",
                                variant="primary",
                                size="lg"
                            )
                    
                    with gr.Row():
                        analysis_summary = gr.HTML(
                            label="📈 Analysis Summary",
                            value="<div style='text-align: center; padding: 40px; color: #6c757d;'>📊 Upload and process a CSV file to see analysis summary</div>"
                        )
                    
                    with gr.Row():
                        download_file = gr.File(
                            label="💾 Download Results",
                            interactive=False
                        )
                
                with gr.Tab("📈 Interactive Visualizations", elem_id="viz-tab"):
                    gr.HTML("<h2 style='text-align: center; color: #495057; margin-bottom: 20px;'>🎨 Data Visualizations & Insights</h2>")
                    
                    viz_btn = gr.Button(
                        "🎨 Generate Beautiful Visualizations",
                        variant="primary",
                        size="lg"
                    )
                    
                    viz_status = gr.Markdown(
                        value="📊 Process a CSV file first, then click above to generate interactive visualizations!"
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            pie_chart = gr.Plot(
                                label="🥧 Sentiment Distribution (Pie Chart)",
                                show_label=True
                            )
                            confidence_chart = gr.Plot(
                                label="📊 Confidence Score Distribution",
                                show_label=True
                            )
                        
                        with gr.Column():
                            bar_chart = gr.Plot(
                                label="📊 Sentiment Distribution (Bar Chart)",
                                show_label=True
                            )
                            wordcloud_display = gr.Image(
                                label="☁️ Word Cloud",
                                show_label=True
                            )
                    
                    with gr.Row():
                        timeline_chart = gr.Plot(
                            label="📈 Sentiment Timeline (if date data available)",
                            show_label=True
                        )
            
            analyze_btn.click(
                fn=self.analyze_single_text,
                inputs=[text_input, single_preprocessing],
                outputs=[sentiment_result, cleaned_text_display, confidence_display]
            )
            
            file_upload.change(
                fn=self._get_csv_columns,
                inputs=[file_upload],
                outputs=[column_dropdown]
            )
            
            process_btn.click(
                fn=self.process_csv_file,
                inputs=[file_upload, column_dropdown, csv_preprocessing],
                outputs=[analysis_summary, download_file, column_dropdown]
            )
            
            viz_btn.click(
                fn=self.generate_visualizations,
                outputs=[pie_chart, bar_chart, confidence_chart, wordcloud_display, timeline_chart, viz_status]
            )
        
        return interface

def main():
    print("🚀 Starting Advanced Sentiment Analysis Tool...")
    print("🎨 Loading AI models and preparing beautiful interface...")
    
    app = SentimentAnalysisApp()
    interface = app.create_interface()
    
    print("✅ Application ready!")
    print("🌐 Opening in your default web browser...")
    print("📱 Access at: http://localhost:7860")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        show_api=False
    )

if __name__ == "__main__":
    main()
