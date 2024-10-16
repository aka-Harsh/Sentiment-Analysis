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

## Video Demo
🎥 Here you can find a video of the working project.

https://github.com/user-attachments/assets/7fe2ed90-ef1c-4c00-b866-7ea4ea641a89

## Prerequisites

Install Python on your system 👉 [Python](https://www.python.org/downloads/).
<br><br>
Please upload a .csv file for the analysis and make sure the **Customer Feedback** column is named **feedback**.

## Deployment

To run this project first clone this repository using:

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

![Screenshot 2024-09-02 154857](https://github.com/user-attachments/assets/ce4c8957-823e-4db1-bbc7-d940343ab374)

![Screenshot 2024-09-02 154958](https://github.com/user-attachments/assets/f44bd895-c97e-4bb7-b8e6-121e973dc2e8)

![Screenshot 2024-09-02 155009](https://github.com/user-attachments/assets/7053430e-5896-481d-bc93-5dd487314bf4)


## FAQ
#### Are you facing this issue (fbgemm.dll [WinError 126] The specified module could not be found)

👇 Paste the libomp140.x86_64.dll file from the repo in 

**C:\Windows\System32**
