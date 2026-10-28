# SentimentX AI

**Machine-Learning-Based Text Sentiment Analysis**

SentimentX AI is a Streamlit-based application that analyzes text and predicts its sentiment as **Positive, Neutral, or Negative**. It supports individual text analysis, CSV-based bulk analysis, visual summaries, session-based analysis history, and report downloads.

## Project Links

- **Live Application:** https://karan-2317-sentiment-analysis-app-ja7qt9.streamlit.app/
- **GitHub Repository:** https://github.com/KARAN-2317/Sentiment_analysis

## Features

- **Dashboard:** Displays project information, available model evaluation metrics, and sentiment distribution visualizations.
- **Text Analyzer:** Predicts the sentiment of individual text and displays available confidence information.
- **Bulk Analyzer:** Processes text records from CSV files and displays prediction results and sentiment summaries.
- **Analysis History:** Reviews analyses saved during the current Streamlit session.
- **Data Visualization:** Presents sentiment distribution through charts.
- **Report Export:** Supports CSV and PDF downloads where provided in the application.
- **Project Information:** Includes About Project and About Me sections.

## Technology Stack

- Python
- Streamlit
- Pandas
- Plotly
- scikit-learn
- TF-IDF text vectorization
- Logistic Regression

## Model Workflow

1. The user enters text or uploads a CSV file.
2. The application prepares the input for analysis.
3. TF-IDF converts text into numerical features.
4. The trained Logistic Regression classifier predicts sentiment.
5. The application displays the prediction and available confidence information.
6. Results can be reviewed, saved in session history, and exported.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/KARAN-2317/Sentiment_analysis.git
cd Sentiment_analysis
```

Install dependencies if `requirements.txt` is available:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Limitations

- Prediction quality depends on the model and its training data.
- Sarcasm and context-dependent language may be difficult to classify correctly.
- Bulk analysis requires a readable CSV with a supported text column.
- Analysis history is stored in the current session and is not permanently saved to a database.

## Academic Purpose

SentimentX AI demonstrates the practical integration of machine learning and an interactive web interface for text sentiment classification.