# Sentiment Analysis
🛠️ This sentiment analysis project uses pre-trained BERT models from Hugging Face Transformers. It can analyze sentiment either from input text or multiple **User Feedback** entries in a CSV file.<br>
<br><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="30" alt="python logo"  />
<img width="12" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" height="30" alt="vscode logo"  />
<img width="12" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" height="30" alt="html5 logo"  />
<img width="12" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" height="30" alt="css3 logo"  />
<img width="12" />
<img src="https://www.pngfind.com/pngs/m/128-1286693_flask-framework-logo-svg-hd-png-download.png" height="30" alt="flask logo"  />
<img width="12" />

## 🆕 What's New in v2.0! 

### 🎭 **Major UI Upgrade - Gradio Interface**
- **Modern Gradio Interface**: Replaced Flask with beautiful, responsive Gradio UI
- **Animated Elements**: Stunning animations, hover effects, and smooth transitions
- **Gradient Themes**: Modern glassmorphism design with beautiful color schemes
- **Tabbed Navigation**: Clean, organized interface with three main sections

### 📁 **Enhanced CSV Processing**
- **Drag & Drop Upload**: Easy file uploading with visual feedback
- **Smart Column Detection**: Automatically detects and suggests text columns
- **Flexible Column Selection**: Choose any column name (not just "feedback")
- **Real-time Preview**: See available columns instantly after upload

### 🧹 **Advanced Text Preprocessing**
- **Automated Text Cleaning**: 
  - URL removal (http, https, www links)
  - Email address removal
  - Social media cleanup (@mentions, #hashtags)
  - Emoji and special character removal
  - Lowercasing and normalization
- **NLP Processing**:
  - Stopword removal
  - Lemmatization for better accuracy
  - Token filtering and optimization
- **Optional Processing**: Toggle preprocessing on/off based on your needs

### 📊 **Interactive Visualizations**
- **Pie Charts**: Beautiful sentiment distribution with custom colors
- **Bar Charts**: Interactive sentiment counts with hover effects
- **Word Clouds**: Visual representation of most common words
- **Confidence Distribution**: Histogram showing prediction reliability
- **Timeline Analysis**: Sentiment trends over time (if date column exists)
- **All charts powered by Plotly** for full interactivity

### 🎯 **Enhanced Analysis Features**
- **Confidence Scores**: Get prediction confidence (0-100%) for every analysis
- **Batch Processing**: Efficient processing of large CSV files with progress tracking
- **Summary Statistics**: Comprehensive analysis summary with key metrics
- **Export Enhanced Results**: Download CSV with original data + sentiment + confidence
- **Error Handling**: Robust error handling with helpful messages

### 🎨 **User Experience Improvements**
- **Progress Indicators**: Real-time progress tracking for file processing
- **Loading Animations**: Beautiful loading states with descriptive messages
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Instant Feedback**: Real-time results and status updates
- **Professional Styling**: Enterprise-level design and animations

## Video Demo
🎥 Here you can find a video of the working project.

https://github.com/user-attachments/assets/7fe2ed90-ef1c-4c00-b866-7ea4ea641a89

## Prerequisites
Install Python on your system 👉 [Python](https://www.python.org/downloads/).
<br><br>
~~Please upload a .csv file for the analysis and make sure the **Customer Feedback** column is named **feedback**.~~ <br>
**📝 NEW: You can now upload any CSV file and select any text column from a dropdown menu! No need to rename columns.**

## Deployment

### 🚀 **Quick Start (New Gradio Version)**
To run the **new enhanced version** with Gradio interface:

```bash
# Clone the repository
git clone https://github.com/aka-Harsh/Sentiment-Analysis.git
cd Sentiment-Analysis

# Create and activate virtual environment
python -m venv sentiment_env
# Windows:
sentiment_env\Scripts\activate
# macOS/Linux:
source sentiment_env/bin/activate

# Install updated requirements
pip install -r requirements.txt

# Run the new Gradio application
python gradio_app.py
```

Open your browser and go to **http://localhost:7860** 🌐

### 🔄 **Legacy Flask Version**
To run the original Flask version:
```bash
  git clone https://github.com/aka-Harsh/Sentiment-Analysis.git
```
Locate this repository using terminal and then create a virtual enviroment and activate it using:
```bash
  python -m venv venv
  .\venv\Scripts\activate
```
Perform this in your VScode editor to select python intepreter:
```bash
  Select View > Command Palette > Python: Select Interpreter > Enter Interpreter path > venv > Script > python.exe
```
Install all the required packages:
```bash
  pip install -r requirements.txt
```
Finally run the app.py file:
```bash
  python app.py
```
Open a web browser and go to http://localhost:5000


## Project Outlook
<br>

### 🆕 **New Gradio Interface Screenshots**
*Beautiful modern interface with animations and interactive elements*

![Image](https://github.com/user-attachments/assets/dcbfa412-926c-4321-bb4f-eb99eb4ff00d)

![Image](https://github.com/user-attachments/assets/c2b9d01f-d471-470d-9d9a-2435891d3005)

![Image](https://github.com/user-attachments/assets/36d73bd1-7196-4e63-a438-a59b0a322202)
![Image](https://github.com/user-attachments/assets/b9143bf2-673a-4af1-a93d-6072f7d6db91)
![Image](https://github.com/user-attachments/assets/bdd2e808-42fb-484c-9dec-052dd0d66ee9)

### **Old Interface Screenshots**
![Screenshot 2024-09-02 154857](https://github.com/user-attachments/assets/ce4c8957-823e-4db1-bbc7-d940343ab374)
![Screenshot 2024-09-02 154958](https://github.com/user-attachments/assets/f44bd895-c97e-4bb7-b8e6-121e973dc2e8)
![Screenshot 2024-09-02 155009](https://github.com/user-attachments/assets/7053430e-5896-481d-bc93-5dd487314bf4)

## 🆕 **New Features Summary**

| Feature | Old Version | New Version |
|---------|-------------|-------------|
| **Interface** | Flask HTML | Modern Gradio with animations |
| **File Upload** | Basic upload | Drag & drop with progress |
| **Column Selection** | Fixed "feedback" | Any column via dropdown |
| **Text Processing** | Basic | Advanced preprocessing pipeline |
| **Visualizations** | None | Interactive Plotly charts |
| **Confidence Scores** | No | Yes, with every prediction |
| **Progress Tracking** | No | Real-time progress indicators |
| **Mobile Support** | Limited | Fully responsive |
| **Error Handling** | Basic | Comprehensive with helpful messages |

## FAQ
#### Are you facing this issue (fbgemm.dll [WinError 126] The specified module could not be found)
👇 Paste the libomp140.x86_64.dll file from the repo in 
**C:\Windows\System32**

#### 🆕 **New FAQ Items**

#### Which version should I use?
- **Use `gradio_app.py`** for the new enhanced experience with modern UI, visualizations, and advanced features
- **Use `app.py`** only if you need the legacy Flask interface
