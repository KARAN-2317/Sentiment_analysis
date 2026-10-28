
import html
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "evaluation.json"
TRAIN_DATA_PATH = BASE_DIR / "data" / "raw" / "train_df.csv"

st.set_page_config(
    page_title="SentimentX AI",
    page_icon="💬",
    layout="wide",
)

SENTIMENT_COLORS = {
    "Positive": "#16A34A",
    "Neutral": "#64748B",
    "Negative": "#DC2626",
}

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]


# =========================================================
# 2. LOAD MODEL AND METRICS
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data
def load_training_distribution():
    if not TRAIN_DATA_PATH.exists():
        return pd.DataFrame({
            "Sentiment": SENTIMENT_ORDER,
            "Examples": [0, 0, 0],
        })

    df = pd.read_csv(TRAIN_DATA_PATH, usecols=["sentiment"])
    counts = (
        df["sentiment"].astype(str).str.lower().value_counts()
    )

    return pd.DataFrame({
        "Sentiment": SENTIMENT_ORDER,
        "Examples": [
            int(counts.get("negative", 0)),
            int(counts.get("neutral", 0)),
            int(counts.get("positive", 0)),
        ],
    })


# =========================================================
# 3. TEXT ANALYSIS
# =========================================================

def analyze_texts(texts, model):
    cleaned = [
        re.sub(r"\s+", " ", str(text)).strip()
        for text in texts
    ]

    results = [
        {
            "Text": str(original),
            "Sentiment": "Not analyzed",
            "Confidence (%)": None,
        }
        for original in texts
    ]

    valid_indices = [
        index for index, text in enumerate(cleaned) if text
    ]

    if not valid_indices:
        return pd.DataFrame(results)

    valid_texts = [cleaned[index] for index in valid_indices]

    predictions = model.predict(valid_texts)
    probabilities = model.predict_proba(valid_texts)

    for index, prediction, probability in zip(
        valid_indices, predictions, probabilities
    ):
        results[index]["Sentiment"] = str(prediction).capitalize()
        results[index]["Confidence (%)"] = round(
            float(max(probability)) * 100, 2
        )

    return pd.DataFrame(results)


def save_to_history(results, source="Text Analyzer"):
    """Save analyzed rows to the current Streamlit session."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in results.iterrows():
        if row["Sentiment"] == "Not analyzed":
            continue

        st.session_state.analysis_history.append({
            "Time": timestamp,
            "Source": source,
            "Text": str(row["Text"]),
            "Sentiment": row["Sentiment"],
            "Confidence (%)": row["Confidence (%)"],
        })


# =========================================================
# 4. PDF REPORT GENERATOR
# =========================================================

def create_pdf_report(
    results,
    title="SentimentX AI Analysis Report",
    source="SentimentX AI",
):
    """Create a downloadable PDF report from analysis results."""
    buffer = BytesIO()

    # Landscape layout gives the text column more room.
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(html.escape(title), styles["Title"]))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Generated: "
            + datetime.now().strftime("%d %b %Y, %I:%M %p"),
            styles["Normal"],
        )
    )
    story.append(Paragraph(f"Report type: {html.escape(source)}", styles["Normal"]))
    story.append(Spacer(1, 14))

    total = len(results)
    valid = results[results["Sentiment"] != "Not analyzed"]

    story.append(
        Paragraph(
            f"Total entries: {total} | Successfully analyzed: {len(valid)}",
            styles["Heading2"],
        )
    )
    story.append(Spacer(1, 8))

    if not valid.empty:
        counts = valid["Sentiment"].value_counts()

        summary_data = [["Sentiment", "Number of entries"]]

        for sentiment in SENTIMENT_ORDER:
            summary_data.append([
                sentiment,
                str(int(counts.get(sentiment, 0))),
            ])

        summary_table = Table(
            summary_data,
            colWidths=[160, 150],
            repeatRows=1,
        )

        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 18))

    story.append(Paragraph("Analysis Details", styles["Heading2"]))
    story.append(Spacer(1, 8))

    body_style = getSampleStyleSheet()["BodyText"]
    body_style.fontSize = 8
    body_style.leading = 10

    table_data = [[
        "Text",
        "Predicted sentiment",
        "Confidence",
    ]]

    for _, row in results.iterrows():
        text = re.sub(r"\s+", " ", str(row["Text"])).strip()

        confidence = row["Confidence (%)"]
        confidence_text = (
            f"{float(confidence):.2f}%"
            if pd.notna(confidence)
            else "N/A"
        )

        table_data.append([
            Paragraph(html.escape(text), body_style),
            Paragraph(html.escape(str(row["Sentiment"])), body_style),
            confidence_text,
        ])

    details_table = Table(
        table_data,
        colWidths=[480, 140, 100],
        repeatRows=1,
        hAlign="LEFT",
    )

    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(details_table)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Important: confidence is the model's estimated probability "
            "for its predicted class. It is not a guarantee of correctness. "
            "Results may vary on different text sources.",
            styles["Italic"],
        )
    )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# 5. INITIALIZE SESSION STATE AND LOAD MODEL
# =========================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

try:
    model = load_model()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Verify that models/sentiment_model.joblib exists. "
        "If it does not, run python src/train_model.py."
    )
    st.stop()

metrics = load_metrics()
test_metrics = metrics.get("test", {})
training_records = metrics.get("training_records", 0)
test_accuracy = test_metrics.get("accuracy")
macro_f1 = test_metrics.get("macro_f1")


# =========================================================
# 6. HEADER AND NAVIGATION
# =========================================================

st.title("💬 SentimentX AI")
st.caption(
    "Sentiment Intelligence Platform | NLP • Machine Learning • Analytics"
)

st.success(
    "Trained TF-IDF + Logistic Regression model loaded successfully."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Text Analyzer",
        "Bulk Analyzer",
        "Analysis History",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("SentimentX AI | Three-class text classification")


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":
    st.header("Sentiment Overview")

    first, second, third = st.columns(3)

    first.metric(
        "Training Records",
        f"{training_records:,}" if training_records else "N/A",
    )

    second.metric(
        "Test Accuracy",
        f"{test_accuracy * 100:.2f}%"
        if test_accuracy is not None
        else "N/A",
    )

    third.metric(
        "Macro F1-Score",
        f"{macro_f1 * 100:.2f}%"
        if macro_f1 is not None
        else "N/A",
    )

    st.caption(
        "Metrics are calculated on the held-out test dataset. "
        "They do not guarantee the same performance on every input."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Training Dataset Distribution")

        distribution = load_training_distribution()

        fig = px.pie(
            distribution,
            names="Sentiment",
            values="Examples",
            title="Training Examples by Sentiment",
            hole=0.45,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Test Set Performance")

        report = test_metrics.get("classification_report", {})
        rows = []

        for sentiment in ["negative", "neutral", "positive"]:
            values = report.get(sentiment, {})

            if values:
                rows.append({
                    "Sentiment": sentiment.capitalize(),
                    "Precision (%)": round(
                        values.get("precision", 0) * 100, 2
                    ),
                    "Recall (%)": round(
                        values.get("recall", 0) * 100, 2
                    ),
                    "F1-Score (%)": round(
                        values.get("f1-score", 0) * 100, 2
                    ),
                })

        if rows:
            performance = pd.DataFrame(rows)

            st.dataframe(
                performance,
                use_container_width=True,
                hide_index=True,
            )

            chart_data = performance.melt(
                id_vars="Sentiment",
                value_vars=[
                    "Precision (%)",
                    "Recall (%)",
                    "F1-Score (%)",
                ],
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                chart_data,
                x="Sentiment",
                y="Score",
                color="Metric",
                barmode="group",
                title="Class-wise Evaluation",
                range_y=[0, 100],
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Evaluation metrics are not available yet.")

    st.divider()
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)
    col1.markdown("**Algorithm**\n\nLogistic Regression")
    col2.markdown("**Text Features**\n\nTF-IDF uni-grams and bi-grams")
    col3.markdown("**Classes**\n\nNegative, Neutral, Positive")


# =========================================================
# 8. TEXT ANALYZER
# =========================================================

elif page == "Text Analyzer":
    st.header("Analyze Your Text")

    text = st.text_area(
        "Enter a review, feedback or social-media comment",
        placeholder=(
            "Example: The camera is excellent "
            "but the battery is poor."
        ),
        height=150,
        key="single_text_input",
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.error("Please enter some text.")
        else:
            result = analyze_texts([text], model)

            save_to_history(result, source="Text Analyzer")

            sentiment = result.iloc[0]["Sentiment"]
            confidence = result.iloc[0]["Confidence (%)"]

            st.subheader(f"Predicted sentiment: {sentiment}")

            col1, col2 = st.columns(2)
            col1.metric("Sentiment", sentiment)
            col2.metric(
                "Estimated model confidence",
                f"{confidence:.2f}%",
            )

            if sentiment == "Positive":
                st.success("The model classified this text as positive.")
            elif sentiment == "Negative":
                st.error("The model classified this text as negative.")
            else:
                st.info("The model classified this text as neutral.")

            st.caption(
                "Confidence is an estimated model probability, "
                "not a guarantee of correctness."
            )

            with st.expander("View all class probabilities"):
                probabilities = model.predict_proba([text.strip()])[0]

                probability_df = pd.DataFrame({
                    "Sentiment": [
                        str(label).capitalize()
                        for label in model.classes_
                    ],
                    "Estimated probability (%)": [
                        round(float(value) * 100, 2)
                        for value in probabilities
                    ],
                })

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download PDF Report",
                data=create_pdf_report(
                    result,
                    title="SentimentX AI Text Analysis Report",
                    source="Single Text Analysis",
                ),
                file_name="sentimentx_analysis.pdf",
                mime="application/pdf",
            )


# =========================================================
# 9. BULK CSV ANALYZER
# =========================================================

elif page == "Bulk Analyzer":
    st.header("Bulk Review Analyzer")

    st.write(
        "Upload a CSV containing a text, review, comment or content column."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="bulk_csv_uploader",
    )

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)

            columns = {
                str(column).strip().lower(): column
                for column in data.columns
            }

            source_column = next(
                (
                    columns[name]
                    for name in ("text", "review", "comment", "content")
                    if name in columns
                ),
                None,
            )

            if source_column is None:
                st.error(
                    "No supported text column found. "
                    "Use a column named text, review, comment or content."
                )

            elif data.empty:
                st.warning("The uploaded CSV is empty.")

            else:
                results = analyze_texts(
                    data[source_column].fillna("").tolist(),
                    model,
                )

                st.subheader("Analysis Results")

                valid = results[
                    results["Sentiment"] != "Not analyzed"
                ]

                first, second, third = st.columns(3)
                first.metric("Total Rows", len(results))
                second.metric("Analyzed", len(valid))
                third.metric(
                    "Skipped Blank Rows",
                    len(results) - len(valid),
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                )

                if not valid.empty:
                    st.subheader("Sentiment Distribution")

                    count_data = (
                        valid["Sentiment"]
                        .value_counts()
                        .reindex(SENTIMENT_ORDER, fill_value=0)
                        .rename_axis("Sentiment")
                        .reset_index(name="Reviews")
                    )

                    fig = px.bar(
                        count_data,
                        x="Sentiment",
                        y="Reviews",
                        title="Predicted Sentiment Distribution",
                        color="Sentiment",
                        color_discrete_map=SENTIMENT_COLORS,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    confidence_data = valid.groupby(
                        "Sentiment", as_index=False
                    )["Confidence (%)"].mean()

                    confidence_data["Confidence (%)"] = (
                        confidence_data["Confidence (%)"].round(2)
                    )

                    st.subheader("Average Estimated Confidence")

                    st.dataframe(
                        confidence_data,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button("Save Bulk Results to History"):
                        save_to_history(
                            results,
                            source="Bulk CSV Analyzer",
                        )
                        st.success(
                            f"Saved {len(valid)} analyzed rows to history."
                        )

                st.download_button(
                    "Download Results CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )

                st.download_button(
                    "Download PDF Report",
                    data=create_pdf_report(
                        results,
                        title="SentimentX AI Bulk Analysis Report",
                        source="Bulk CSV Analysis",
                    ),
                    file_name="sentimentx_bulk_report.pdf",
                    mime="application/pdf",
                )

        except (ValueError, UnicodeDecodeError) as error:
            st.error(f"Unable to read CSV: {error}")


# =========================================================
# 10. ANALYSIS HISTORY
# =========================================================

elif page == "Analysis History":
    st.header("Analysis History")

    history = st.session_state.get("analysis_history", [])

    if history:
        history_df = pd.DataFrame(history)

        first, second, third = st.columns(3)

        first.metric("Saved Analyses", len(history_df))
        second.metric(
            "Positive",
            int((history_df["Sentiment"] == "Positive").sum()),
        )
        third.metric(
            "Negative",
            int((history_df["Sentiment"] == "Negative").sum()),
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="sentimentx_history.csv",
            mime="text/csv",
        )

        pdf_history = history_df[
            ["Text", "Sentiment", "Confidence (%)"]
        ]

        st.download_button(
            "Download History PDF",
            data=create_pdf_report(
                pdf_history,
                title="SentimentX AI Analysis History",
                source="Current Session History",
            ),
            file_name="sentimentx_history.pdf",
            mime="application/pdf",
        )

        st.caption(
            "History is stored in the current Streamlit session. "
            "It is not yet saved permanently to a database."
        )

        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()

    else:
        st.info(
            "No analysis history yet. Analyze text or save bulk results "
            "to see them here."
        )


# =========================================================
# 11. ABOUT PROJECT
# =========================================================

elif page == "About Project":
    st.header("About SentimentX AI")

    st.write(
        "SentimentX AI is a Natural Language Processing application "
        "that classifies text into positive, negative and neutral sentiment."
    )

    st.subheader("Current methodology")

    st.markdown(
        """
        1. **Input:** Accept a review, feedback message or social-media text.
        2. **Text features:** TF-IDF converts text into numerical features
           using individual words and two-word combinations.
        3. **Classification:** Logistic Regression predicts one of the
           three sentiment classes.
        4. **Estimated probability:** The application displays estimated
           class probabilities.
        5. **Batch analysis:** CSV files can be analyzed and exported.
        6. **PDF reporting:** Export individual or bulk analysis results.
        7. **Session history:** Review analyses saved during the current session.
        8. **Evaluation:** Display test-set accuracy, precision, recall and F1.
        """
    )

    st.subheader("Training and evaluation")

    if training_records:
        st.write(f"Training records: {training_records:,}")

    if test_accuracy is not None:
        st.write(f"Test accuracy: {test_accuracy * 100:.2f}%")

    if macro_f1 is not None:
        st.write(f"Test macro F1-score: {macro_f1 * 100:.2f}%")

    st.subheader("Current limitations")

    st.markdown(
        """
        - Sarcasm, slang, irony and mixed opinions may be misclassified.
        - Neutral text can be difficult to distinguish from other classes.
        - Confidence is not a guarantee of prediction correctness.
        - Evaluation results apply to the provided test dataset and may
          differ on other text sources.
        - Emotion detection and aspect-based sentiment analysis are
          not implemented in this version.
        - Analysis history currently lasts only for the current session.
        """
    )

    st.caption(
        "Model: TF-IDF + Logistic Regression | "
        "Classes: Negative, Neutral and Positive"
    )
